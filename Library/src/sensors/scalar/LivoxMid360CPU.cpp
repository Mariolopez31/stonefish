#include "sensors/scalar/LivoxMid360CPU.h"

#include "core/SimulationApp.h"
#include "core/SimulationManager.h"
#include "btBulletDynamicsCommon.h"

#include <cmath>
#include <algorithm>

#define DEG2RAD(x) ((x) * SIMD_PI / Scalar(180.0))

namespace sf
{

struct ClosestRaySkipList : btCollisionWorld::ClosestRayResultCallback
{
  const std::vector<const btCollisionObject*>& skip;

  ClosestRaySkipList(const btVector3& from, const btVector3& to,
                     const std::vector<const btCollisionObject*>& s)
  : btCollisionWorld::ClosestRayResultCallback(from, to), skip(s) {}

  bool needsCollision(btBroadphaseProxy* proxy0) const override
  {
    if(!btCollisionWorld::ClosestRayResultCallback::needsCollision(proxy0)) return false;

    const btCollisionObject* obj = static_cast<const btCollisionObject*>(proxy0->m_clientObject);
    for(const auto* s : skip)
      if(s == obj) return false;

    return true;
  }
};

static inline uint8_t clamp_u8(int v)
{
  return (uint8_t)std::max(0, std::min(255, v));
}

LivoxMid360CPU::LivoxMid360CPU(std::string name,
                               unsigned int horizontalRes,
                               unsigned int verticalRes,
                               Scalar minRange,
                               Scalar maxRange,
                               Scalar frameRateHz)
: LinkSensor(name, frameRateHz, 1),
  hRes_(horizontalRes), vRes_(verticalRes),
  rangeMin_(minRange), rangeMax_(maxRange),
  frameHz_(frameRateHz)
{
  // mitad de columnas -> mitad de puntos
  hStep_ = 2;
  if(hStep_ < 1) hStep_ = 1;

  setFovDeg(Scalar(360.0), Scalar(-7.0), Scalar(52.0));
  setPointRate(Scalar(200000.0));
  setLines(4);
  setNonRepetitive(true);

  buildRayLUT_(Scalar(0.0));
  points_.reserve(rayDirsLocal_.size());
}

void LivoxMid360CPU::setFovDeg(Scalar fovH_deg, Scalar vMin_deg, Scalar vMax_deg)
{
  fovHdeg_ = fovH_deg;
  vMinDeg_ = vMin_deg;
  vMaxDeg_ = vMax_deg;
}

void LivoxMid360CPU::setPointRate(Scalar pts_per_sec) { pointRate_ = btMax(pts_per_sec, Scalar(1.0)); }
void LivoxMid360CPU::setLines(uint8_t lines) { lines_ = (lines == 0) ? 1 : lines; }
void LivoxMid360CPU::setNonRepetitive(bool on) { nonRepetitive_ = on; }
void LivoxMid360CPU::setLidarId(uint8_t id) { lidarId_ = id; }

void LivoxMid360CPU::buildRayLUT_(Scalar yawOffsetDeg)
{
  rayDirsLocal_.clear();
  rayIdxToGlobal_.clear();

  // reserva aprox: (vRes * hRes/step), luego se recorta por sp<0
  rayDirsLocal_.reserve(size_t(vRes_) * size_t((hRes_ + hStep_ - 1) / hStep_));
  rayIdxToGlobal_.reserve(rayDirsLocal_.capacity());

  const Scalar vRange = (vMaxDeg_ - vMinDeg_);
  const Scalar vStep  = (vRes_ > 1) ? (vRange / Scalar(vRes_ - 1)) : Scalar(0.0);

  const Scalar hStart = -fovHdeg_ / Scalar(2.0) + yawOffsetDeg;
  const Scalar hStepDeg = fovHdeg_ / Scalar(hRes_);

  std::vector<Scalar> cy(hRes_), sy(hRes_);
  for(unsigned int j=0; j<hRes_; ++j)
  {
    const Scalar yaw = DEG2RAD(hStart + Scalar(j) * hStepDeg);
    cy[j] = btCos(yaw);
    sy[j] = btSin(yaw);
  }

  for(unsigned int i=0; i<vRes_; ++i)
  {
    const Scalar pitch = DEG2RAD(vMinDeg_ + Scalar(i) * vStep);
    const Scalar cp = btCos(pitch);
    const Scalar sp = btSin(pitch);

    // descarta rayos hacia abajo (reduce carga + evita agua por debajo)
    if(sp < Scalar(0.0))
      continue;

    for(unsigned int j=0; j<hRes_; j += hStep_)
    {
      rayDirsLocal_.push_back(Vector3(cp*cy[j], cp*sy[j], sp));
      rayIdxToGlobal_.push_back(uint32_t(i*hRes_ + j));
    }
  }
}

void LivoxMid360CPU::rebuildSkipList_(btDynamicsWorld* world, const Vector3& sensorPos)
{
  skip_.clear();
  if(!world) return;

  // como tu LidarCPU: quita dinámicos cerca (barco)
  const Scalar selfR  = Scalar(1.2);
  const Scalar selfR2 = selfR*selfR;

  const int n = world->getNumCollisionObjects();
  for(int k=0; k<n; ++k)
  {
    const btCollisionObject* obj = world->getCollisionObjectArray()[k];
    if(!obj) continue;

    if(obj->isStaticOrKinematicObject())
      continue;

    const btVector3 op = obj->getWorldTransform().getOrigin();
    const Scalar dx = Scalar(op.x()) - sensorPos.x();
    const Scalar dy = Scalar(op.y()) - sensorPos.y();
    const Scalar dz = Scalar(op.z()) - sensorPos.z();
    const Scalar d2 = dx*dx + dy*dy + dz*dz;

    if(d2 < selfR2)
      skip_.push_back(obj);
  }
}

void LivoxMid360CPU::InternalUpdate(Scalar dt)
{
  simTimeNs_ += (uint64_t)llround(double(dt) * 1e9);

  SimulationManager* sim = SimulationApp::getApp()->getSimulationManager();
  if(!sim) return;

  btDynamicsWorld* world = sim->getDynamicsWorld();
  if(!world) return;

  if(rayDirsLocal_.empty())
    return;

  const unsigned int totalRays = (unsigned int)rayDirsLocal_.size();

  if(scanCursor_ == 0)
  {
    points_.clear();

    Scalar yawOff = Scalar(0.0);
    if(nonRepetitive_)
      yawOff = Scalar(std::fmod(double(frameIndex_) * 37.0, 1.0) * double(fovHdeg_ / double(hRes_)));

    buildRayLUT_(yawOff);

    scanTf_     = getSensorFrame();
    scanBasis_  = scanTf_.getBasis();
    scanBasisT_ = scanBasis_.transpose();

    scanOrigin_ = scanTf_.getOrigin() + scanBasis_ * Vector3(Scalar(0.0), Scalar(0.0), Scalar(0.047));
    scanTimebaseNs_ = simTimeNs_;

    if((skipRebuildCounter_++ % 15u) == 0u || skip_.empty())
      rebuildSkipList_(world, scanTf_.getOrigin());
  }

  const double raysPerSec = double(totalRays) * double(frameHz_);
  raysAcc_ += raysPerSec * double(dt);

  unsigned int raysThisUpdate = (unsigned int)std::floor(raysAcc_);
  if(raysThisUpdate == 0) return;
  raysAcc_ -= double(raysThisUpdate);

  // CLAVE: igual que tu LidarCPU
  const Scalar muzzle     = Scalar(0.45);
  const Scalar startRange = btMax(rangeMin_, muzzle);

  const double dtRayNs = 1e9 / double(pointRate_);

  const Scalar waterZ = Scalar(0.0);
  const Scalar waterEps = Scalar(0.03);

  for(unsigned int c=0; c<raysThisUpdate && scanCursor_ < totalRays; ++c)
  {
    const uint32_t gidx = rayIdxToGlobal_[scanCursor_];
    const Vector3 dirW = scanBasis_ * rayDirsLocal_[scanCursor_];

    const Vector3 start = scanOrigin_ + dirW * startRange;
    const Vector3 end   = scanOrigin_ + dirW * rangeMax_;

    ClosestRaySkipList rayCallback(start, end, skip_);
    rayCallback.m_collisionFilterGroup = -1;
    rayCallback.m_collisionFilterMask  = -1;

    world->rayTest(start, end, rayCallback);

    if(rayCallback.hasHit())
    {
      const Vector3 hitW = rayCallback.m_hitPointWorld;

      // filtra agua (como tu LidarCPU)
      if(btFabs(hitW.z() - waterZ) >= waterEps)
      {
        const Scalar dist = startRange + rayCallback.m_closestHitFraction * (rangeMax_ - startRange);

        if(dist >= rangeMin_ && dist <= rangeMax_)
        {
          const Vector3 pL = scanBasisT_ * (hitW - scanTf_.getOrigin());

          const unsigned int i = unsigned(gidx / hRes_);
          const uint8_t line = (uint8_t)std::min<int>(
              int(lines_ - 1),
              int((uint64_t(i) * lines_) / std::max(1u, vRes_)));

          LivoxPoint p;
          p.offset_time = (uint32_t)std::min<uint64_t>(
              (uint64_t)llround(dtRayNs * double(gidx)), 0xFFFFFFFFu);

          p.x = (float)pL.x();
          p.y = (float)pL.y();
          p.z = (float)pL.z();

          const double refl =
              (1.0 - std::min(1.0, std::max(0.0, double(dist / rangeMax_)))) * 255.0;
          p.reflectivity = clamp_u8((int)llround(refl));

          p.tag  = 0;
          p.line = line;

          points_.push_back(p);
        }
      }
    }

    ++scanCursor_;
  }

  if(scanCursor_ >= totalRays)
  {
    scanCursor_ = 0;
    frameIndex_++;

    if(callback_) callback_(scanTimebaseNs_, lidarId_, points_);
  }
}

} // namespace sf