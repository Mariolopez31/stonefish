#include "sensors/scalar/Lidar.h"
#include "core/SimulationApp.h"
#include "core/SimulationManager.h"
#include "btBulletDynamicsCommon.h"
#include "graphics/OpenGLPipeline.h"

#include <glm/vec3.hpp>
#include <vector>
#include <chrono>
#include <memory>
#include <mutex>
#include <cmath>
#include <iostream>   // <- FIX

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

Lidar::Lidar(std::string name, unsigned int horizontalRes, unsigned int verticalRes,
             Scalar fovH, Scalar fovV, Scalar minRange, Scalar maxRange, Scalar frequency)
  : LinkSensor(name, frequency, 1),
    hRes(horizontalRes), vRes(verticalRes),
    fovH(fovH), fovV(fovV),
    rangeMin(minRange), rangeMax(maxRange),
    freqHz_(frequency)
{
  bufferPoints.reserve(hRes * vRes);
  callback_ = nullptr;

  buildRayLUT_();
  if (debugRender) buildDebugLUT_();
}

Lidar::~Lidar() {}

SensorType Lidar::getType() const { return SensorType::LIDAR; }
ScalarSensorType Lidar::getScalarSensorType() const { return ScalarSensorType::LIDAR; }
void Lidar::setResponseCallback(LidarCallback cb) { callback_ = cb; }

void Lidar::buildRayLUT_()
{
  rayDirsLocal_.assign(size_t(vRes) * size_t(hRes), Vector3(0,0,0));

  const Scalar vMinDegrees = Scalar(0.5);
  const Scalar vMaxDegrees = Scalar(52.0);

  const Scalar vStart = vMinDegrees;
  const Scalar vRange = vMaxDegrees - vMinDegrees;
  const Scalar vStep  = (vRes > 1) ? (vRange / Scalar(vRes - 1)) : Scalar(0.0);

  const Scalar hStart = -fovH / Scalar(2.0);
  const Scalar hStep  =  fovH / Scalar(hRes);

  std::vector<Scalar> cy(hRes), sy(hRes);
  for(unsigned int j = 0; j < hRes; ++j)
  {
    const Scalar yaw = DEG2RAD(hStart + Scalar(j) * hStep);
    cy[j] = btCos(yaw);
    sy[j] = btSin(yaw);
  }

  for(unsigned int i = 0; i < vRes; ++i)
  {
    const Scalar pitch = DEG2RAD(vStart + Scalar(i) * vStep);
    const Scalar cp = btCos(pitch);
    const Scalar sp = btSin(pitch);

    for(unsigned int j = 0; j < hRes; ++j)
      rayDirsLocal_[size_t(i) * size_t(hRes) + j] = Vector3(cp * cy[j], cp * sy[j], sp);
  }
}

void Lidar::buildDebugLUT_()
{
  debugLinesWrite.clear();

  const Scalar vMinDegrees = Scalar(0.5);
  const Scalar vMaxDegrees = Scalar(52.0);

  const Scalar muzzle     = Scalar(0.45);
  const Scalar startRange = btMax(rangeMin, muzzle);

  const Scalar rNear = btMax(startRange, Scalar(0.1));
  const Scalar rFar  = btMin(rangeMax,   Scalar(5.0));

  const Scalar pitchMin = DEG2RAD(vMinDegrees);
  const Scalar pitchMax = DEG2RAD(vMaxDegrees);

  const int N = 24;

  auto addLine = [&](const Vector3& a, const Vector3& b)
  {
    debugLinesWrite.push_back(a);
    debugLinesWrite.push_back(b);
  };

  auto ringPoint = [&](Scalar yaw, Scalar pitch, Scalar r)
  {
    Scalar cp = btCos(pitch), sp = btSin(pitch);
    return Vector3(cp * btCos(yaw), cp * btSin(yaw), sp) * r;
  };

  for (int k = 0; k < N; ++k)
  {
    Scalar yaw1 = DEG2RAD(-180.0 + 360.0 * Scalar(k) / Scalar(N));
    Scalar yaw2 = DEG2RAD(-180.0 + 360.0 * Scalar(k + 1) / Scalar(N));

    addLine(ringPoint(yaw1, pitchMin, rFar),  ringPoint(yaw2, pitchMin, rFar));
    addLine(ringPoint(yaw1, pitchMax, rFar),  ringPoint(yaw2, pitchMax, rFar));

    addLine(ringPoint(yaw1, pitchMin, rNear), ringPoint(yaw2, pitchMin, rNear));
    addLine(ringPoint(yaw1, pitchMax, rNear), ringPoint(yaw2, pitchMax, rNear));
  }

  Scalar yaws[4] = { DEG2RAD(-180.0), DEG2RAD(-90.0), DEG2RAD(0.0), DEG2RAD(90.0) };
  for (int m = 0; m < 4; ++m)
  {
    Scalar yaw = yaws[m];
    addLine(ringPoint(yaw, pitchMin, rNear), ringPoint(yaw, pitchMin, rFar));
    addLine(ringPoint(yaw, pitchMax, rNear), ringPoint(yaw, pitchMax, rFar));
  }

  {
    std::lock_guard<std::mutex> lk(debugMutex);
    debugLinesRender = debugLinesWrite;
  }
}

bool Lidar::isWaterCandidate_(const btCollisionObject* obj, Scalar waterZ) const
{
  if(!obj) return false;
  if(!obj->isStaticOrKinematicObject()) return false;

  const btCollisionShape* shape = obj->getCollisionShape();
  if(!shape) return false;

  const int st = shape->getShapeType();

  const bool shapeOk =
      (st == TRIANGLE_MESH_SHAPE_PROXYTYPE) ||
      (st == SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE) ||
      (st == STATIC_PLANE_PROXYTYPE) ||
      (st == TERRAIN_SHAPE_PROXYTYPE);

  if(!shapeOk) return false;

  btVector3 aabbMin, aabbMax;
  shape->getAabb(obj->getWorldTransform(), aabbMin, aabbMax);

  const Scalar sx = Scalar(aabbMax.x() - aabbMin.x());
  const Scalar sy = Scalar(aabbMax.y() - aabbMin.y());
  const Scalar sz = Scalar(aabbMax.z() - aabbMin.z());
  const Scalar zc = Scalar(0.5) * (Scalar(aabbMax.z()) + Scalar(aabbMin.z()));

  if(sx < Scalar(6.0) || sy < Scalar(6.0)) return false;
  if(sz > Scalar(1.0)) return false;
  if(btFabs(zc - waterZ) > Scalar(1.0)) return false;

  return true;
}

void Lidar::rebuildSkipList_(btDynamicsWorld* world, const Vector3& sensorPos)
{
  skip_.clear();
  if(!world) return;

  const Scalar waterZ = Scalar(0.0);

  const Scalar selfR  = Scalar(1.2);
  const Scalar selfR2 = selfR * selfR;

  const int n = world->getNumCollisionObjects();
  for(int k = 0; k < n; ++k)
  {
    const btCollisionObject* obj = world->getCollisionObjectArray()[k];
    if(!obj) continue;

    bool add = false;

    if(obj->isStaticOrKinematicObject())
    {
      if(isWaterCandidate_(obj, waterZ))
        add = true;
    }
    else
    {
      const btVector3 op = obj->getWorldTransform().getOrigin();
      const Scalar dx = Scalar(op.x()) - sensorPos.x();
      const Scalar dy = Scalar(op.y()) - sensorPos.y();
      const Scalar dz = Scalar(op.z()) - sensorPos.z();
      const Scalar d2 = dx*dx + dy*dy + dz*dz;

      if(d2 < selfR2)
        add = true;
    }

    if(add)
    {
      bool already = false;
      for(const auto* s : skip_)
        if(s == obj) { already = true; break; }

      if(!already)
        skip_.push_back(obj);
    }
  }
}

void Lidar::InternalUpdate(Scalar dt)
{
  SimulationManager* sim = SimulationApp::getApp()->getSimulationManager();
  if(!sim) return;

  btDynamicsWorld* world = sim->getDynamicsWorld();
  if(!world) return;

  const unsigned int totalRays = hRes * vRes;

  if(scanCursor_ == 0)
  {
    bufferPoints.clear();

    scanTf_     = getSensorFrame();
    scanBasis_  = scanTf_.getBasis();
    scanBasisT_ = scanBasis_.transpose();

    scanOrigin_ = scanTf_.getOrigin() + scanBasis_ * Vector3(Scalar(0.0), Scalar(0.0), Scalar(0.047));

    if((skipRebuildCounter_++ % 15u) == 0u || skip_.empty())
      rebuildSkipList_(world, scanTf_.getOrigin());
  }

  const double raysPerSec = double(totalRays) * double(freqHz_);
  raysAcc_ += raysPerSec * double(dt);

  unsigned int raysThisUpdate = (unsigned int)std::floor(raysAcc_);
  if(raysThisUpdate == 0) return;
  raysAcc_ -= double(raysThisUpdate);

  const Scalar muzzle     = Scalar(0.45);
  const Scalar startRange = btMax(rangeMin, muzzle);

  for(unsigned int c = 0; c < raysThisUpdate && scanCursor_ < totalRays; ++c)
  {
    const Vector3 worldDir = scanBasis_ * rayDirsLocal_[scanCursor_];

    const Vector3 start = scanOrigin_ + (worldDir * startRange);
    const Vector3 end   = scanOrigin_ + (worldDir * rangeMax);

    ClosestRaySkipList rayCallback(start, end, skip_);
    rayCallback.m_collisionFilterGroup = -1;
    rayCallback.m_collisionFilterMask  = -1;

    world->rayTest(start, end, rayCallback);

    if(rayCallback.hasHit())
    {
      const Vector3 hitPoint = rayCallback.m_hitPointWorld;

      const Scalar waterZ = Scalar(0.0);
      const Scalar eps    = Scalar(0.03);
      if(btFabs(hitPoint.z() - waterZ) >= eps)
      {
        const Scalar dist = startRange + rayCallback.m_closestHitFraction * (rangeMax - startRange);
        if(dist >= rangeMin && dist <= rangeMax)
        {
          const Vector3 localPoint = scanBasisT_ * (hitPoint - scanTf_.getOrigin());

          LidarPoint p;
          p.x = (float)localPoint.x();
          p.y = (float)localPoint.y();
          p.z = (float)localPoint.z();
          p.intensity = (float)(1.0 - (dist / rangeMax));
          bufferPoints.push_back(p);
        }
      }
    }

    ++scanCursor_;
  }

  if(scanCursor_ >= totalRays)
  {
    std::cout << "[LidarCPU] scan done: points=" << bufferPoints.size()
              << " skip=" << skip_.size()
              << " startRange=" << double(startRange)
              << " rangeMin=" << double(rangeMin)
              << " rangeMax=" << double(rangeMax)
              << "\n";

    scanCursor_ = 0;
    if(callback_) callback_(bufferPoints);
  }
}

std::vector<Renderable> Lidar::Render()
{
  std::vector<Renderable> items = LinkSensor::Render();

  if (isRenderable() && debugRender)
  {
    Renderable item;
    item.type  = RenderableType::SENSOR_LINES;
    item.model = glMatrixFromTransform(getSensorFrame());
    item.data  = std::make_shared<std::vector<glm::vec3>>();

    auto points = item.getDataAsPoints();
    {
      std::lock_guard<std::mutex> lk(debugMutex);
      points->reserve(points->size() + debugLinesRender.size());
      for(const auto& p : debugLinesRender)
        points->push_back(glm::vec3((float)p.x(), (float)p.y(), (float)p.z()));
    }

    items.push_back(item);
  }

  return items;
}

} // namespace sf
