#include "sensors/scalar/LidarGPU.h"

#include "Stonefish/core/SimulationApp.h"
#include "Stonefish/core/SimulationManager.h"

#include "core/GraphicalSimulationApp.h"
#include "graphics/OpenGLPipeline.h"
#include "graphics/OpenGLContent.h"
#include "graphics/OpenGLDepthCamera.h"
#include "graphics/OpenGLView.h"
#include "sensors/vision/Camera.h"

#include <glm/vec3.hpp>
#include <cstring>
#include <memory>

#define DEG2RAD(x) ((x) * SIMD_PI / Scalar(180.0))

namespace sf
{

// Receptor mínimo que OpenGLDepthCamera usa para volcar el buffer
struct LidarGPU::GpuReceiver final : public sf::Camera
{
  LidarGPU* owner = nullptr;
  unsigned int seg = 0;

  sf::OpenGLView* view_ = nullptr;

  GpuReceiver(LidarGPU* o, unsigned int s, unsigned int width, unsigned int height, Scalar fovXdeg)
  : sf::Camera(o->getName() + "_gpu_seg" + std::to_string(s),
               width, height, fovXdeg, Scalar(-1.0))
  {
    owner = o;
    seg = s;
    setDisplayOnScreen(false, 0u, 0u, 1.f);
  }

  void setView(sf::OpenGLView* v) { view_ = v; }

  VisionSensorType getVisionSensorType() const override { return VisionSensorType::DEPTH_CAMERA; }
  sf::OpenGLView* getOpenGLView() const override { return view_; }

  void SetupCamera(const Vector3&, const Vector3&, const Vector3&) override {}
  void* getImageDataPointer(unsigned int) override { return nullptr; }

  void NewDataReady(void* data, unsigned int) override
  {
    owner->onRanges_(seg, reinterpret_cast<const float*>(data));
  }

protected:
  void InternalUpdate(Scalar) override {}
  void InitGraphics() override {}
};

LidarGPU::LidarGPU(std::string name,
                   unsigned int horizontalRes,
                   unsigned int verticalRes,
                   Scalar horizontalFovDeg,
                   Scalar verticalFovDeg,
                   Scalar minRange,
                   Scalar maxRange,
                   Scalar frequency)
: LinkSensor(name, frequency, 1),
  hRes_(horizontalRes), vRes_(verticalRes),
  fovHdeg_(horizontalFovDeg), fovVdeg_(verticalFovDeg),
  rangeMin_(minRange), rangeMax_(maxRange),
  freqHz_(frequency)
{
  points_.reserve(size_t(hRes_) * size_t(vRes_));
  if(debugRender_) buildDebug_();
}

LidarGPU::~LidarGPU() {}

SensorType LidarGPU::getType() const { return SensorType::LIDAR; }
ScalarSensorType LidarGPU::getScalarSensorType() const { return ScalarSensorType::LIDAR; }

void LidarGPU::setResponseCallback(LidarCallback cb) { callback_ = cb; }

void LidarGPU::buildDebug_()
{
  std::vector<Vector3> lines;

  const Scalar muzzle     = Scalar(0.10);
  const Scalar startRange = btMax(rangeMin_, muzzle);
  const Scalar rNear = btMax(startRange, Scalar(0.1));
  const Scalar rFar  = btMin(rangeMax_,   Scalar(5.0));

  const Scalar pitchMin = DEG2RAD(vMinDeg_);
  const Scalar pitchMax = DEG2RAD(vMaxDeg_);

  const int N = 24;

  auto addLine = [&](const Vector3& a, const Vector3& b){ lines.push_back(a); lines.push_back(b); };
  auto ringPoint = [&](Scalar yaw, Scalar pitch, Scalar r)
  {
    Scalar cp = btCos(pitch), sp = btSin(pitch);
    return Vector3(cp * btCos(yaw), cp * btSin(yaw), sp) * r;
  };

  for (int k = 0; k < N; ++k)
  {
    Scalar yaw1 = DEG2RAD(-180.0 + 360.0 * Scalar(k) / Scalar(N));
    Scalar yaw2 = DEG2RAD(-180.0 + 360.0 * Scalar(k+1) / Scalar(N));
    addLine(ringPoint(yaw1, pitchMin, rFar), ringPoint(yaw2, pitchMin, rFar));
    addLine(ringPoint(yaw1, pitchMax, rFar), ringPoint(yaw2, pitchMax, rFar));
    addLine(ringPoint(yaw1, pitchMin, rNear), ringPoint(yaw2, pitchMin, rNear));
    addLine(ringPoint(yaw1, pitchMax, rNear), ringPoint(yaw2, pitchMax, rNear));
  }

  Scalar yaws[4] = { DEG2RAD(-180.0), DEG2RAD(-90.0), DEG2RAD(0.0), DEG2RAD(90.0) };
  for(int m=0;m<4;++m)
  {
    Scalar yaw = yaws[m];
    addLine(ringPoint(yaw, pitchMin, rNear), ringPoint(yaw, pitchMin, rFar));
    addLine(ringPoint(yaw, pitchMax, rNear), ringPoint(yaw, pitchMax, rFar));
  }

  std::lock_guard<std::mutex> lk(debugMutex_);
  debugLines_.swap(lines);
}

void LidarGPU::initGpu_()
{
  if(gpuInit_) return;

  auto* base = SimulationApp::getApp();
  auto* gapp = dynamic_cast<GraphicalSimulationApp*>(base);
  if(!gapp || !gapp->hasGraphics() || gapp->getGLPipeline() == nullptr)
    return; // sin GPU

  auto* content = gapp->getGLPipeline()->getContent();

  segs_ = 4;
  allMask_ = (1u << segs_) - 1u;

  segW_.assign(segs_, hRes_ / segs_);
  segW_[segs_ - 1] += (hRes_ % segs_);

  views_.assign(segs_, nullptr);

  // ---- IMPORTANTE: inicializa ranges_ aquí (antes de usar ranges_[s]) ----
  ranges_.clear();
  ranges_.resize(segs_);

  rx_.clear();
  rx_.reserve(segs_);

  const Scalar segFovDeg = fovHdeg_ / Scalar(segs_);

  const Scalar vFovUsedDeg = (vMaxDeg_ - vMinDeg_);
  const Scalar pitchMid = DEG2RAD((vMinDeg_ + vMaxDeg_) * Scalar(0.5));

  const Scalar muzzle = Scalar(0.10);
  const Scalar nearClip = btMax(rangeMin_, muzzle);

  for(unsigned int s=0;s<segs_;++s)
  {
    const unsigned int w = segW_[s];
    ranges_[s].assign(size_t(w) * size_t(vRes_), 0.f);

    auto* view = new OpenGLDepthCamera(
      glm::vec3(0,0,0),
      glm::vec3(1,0,0),
      glm::vec3(0,0,1),
      0, 0,
      (GLint)w, (GLint)vRes_,
      (GLfloat)segFovDeg,
      (GLfloat)nearClip,
      (GLfloat)rangeMax_,
      false, true, (GLfloat)vFovUsedDeg
    );

    auto rx = std::make_unique<GpuReceiver>(this, s, w, vRes_, segFovDeg);

    // <- clave: enlaza el view en el receiver para cumplir getOpenGLView()
    rx->setView(view);

    view->setCamera(rx.get(), s);
    content->AddView(view);

    views_[s] = view;
    rx_.push_back(std::move(rx));
  }

  // LUT dirs (sensor frame)
  dirLUT_.assign(size_t(hRes_) * size_t(vRes_), Vector3(0,0,0));

  const Scalar fovX = DEG2RAD(segFovDeg);
  const Scalar fovY = DEG2RAD(vFovUsedDeg);
  const Scalar tanX = btTan(fovX * Scalar(0.5));
  const Scalar tanY = btTan(fovY * Scalar(0.5));

  unsigned int j0 = 0;
  for(unsigned int s=0;s<segs_;++s)
  {
    const unsigned int w = segW_[s];
    const Scalar yawC = DEG2RAD(-fovHdeg_/Scalar(2.0) + segFovDeg*(Scalar(s)+Scalar(0.5)));

    const Scalar cp = btCos(pitchMid);
    const Scalar sp = btSin(pitchMid);
    const Scalar cy = btCos(yawC);
    const Scalar sy = btSin(yawC);

    Vector3 fwd(cp*cy, cp*sy, sp);
    Vector3 up(-sp*cy, -sp*sy, cp);
    Vector3 right = fwd.cross(up);
    right.normalize(); up.normalize(); fwd.normalize();

    for(unsigned int i=0;i<vRes_;++i)
    {
      const Scalar yN = ((Scalar(i)+Scalar(0.5))/Scalar(vRes_))*Scalar(2.0)-Scalar(1.0);
      for(unsigned int u=0;u<w;++u)
      {
        const Scalar xN = ((Scalar(u)+Scalar(0.5))/Scalar(w))*Scalar(2.0)-Scalar(1.0);
        Vector3 d = fwd + right*(xN*tanX) + up*(yN*tanY);
        d.normalize();
        const unsigned int j = j0 + u;
        dirLUT_[size_t(i)*size_t(hRes_) + size_t(j)] = d;
      }
    }

    j0 += w;
  }

  readyMask_.store(0u);
  gpuInit_ = true;
}

void LidarGPU::onRanges_(unsigned int seg, const float* src)
{
  if(seg >= segs_) return;

  const size_t n = ranges_[seg].size();
  {
    // ---- mutex global (NO vector de mutex) ----
    std::lock_guard<std::mutex> lk(rangesMtx_);
    std::memcpy(ranges_[seg].data(), src, n * sizeof(float));
  }
  readyMask_.fetch_or(1u << seg, std::memory_order_release);
}

void LidarGPU::requestScan_()
{
  scanTf_   = getSensorFrame();
  basisW_   = scanTf_.getBasis();
  basisWT_  = basisW_.transpose();

  originW_ = scanTf_.getOrigin() + basisW_ * Vector3(Scalar(0.0), Scalar(0.0), Scalar(0.047));

  const glm::vec3 eye((float)originW_.x(), (float)originW_.y(), (float)originW_.z());
  const Scalar segFovDeg = fovHdeg_ / Scalar(segs_);

  const Scalar vFovUsedDeg = (vMaxDeg_ - vMinDeg_);
  const Scalar pitchMid = DEG2RAD((vMinDeg_ + vMaxDeg_) * Scalar(0.5));

  readyMask_.store(0u, std::memory_order_release);

  for(unsigned int s=0;s<segs_;++s)
  {
    const Scalar yawC = DEG2RAD(-fovHdeg_/Scalar(2.0) + segFovDeg*(Scalar(s)+Scalar(0.5)));

    const Scalar cp = btCos(pitchMid);
    const Scalar sp = btSin(pitchMid);
    const Scalar cy = btCos(yawC);
    const Scalar sy = btSin(yawC);

    Vector3 fwdL(cp*cy, cp*sy, sp);
    Vector3 upL(-sp*cy, -sp*sy, cp);

    Vector3 fwdW = basisW_ * fwdL;
    Vector3 upW  = basisW_ * upL;

    const glm::vec3 dir((float)fwdW.x(), (float)fwdW.y(), (float)fwdW.z());
    const glm::vec3 up ((float)upW.x(),  (float)upW.y(),  (float)upW.z());

    views_[s]->SetupCamera(eye, dir, up);
    views_[s]->Update();
  }

  scanRequested_ = true;
}

void LidarGPU::buildPointCloud_()
{
  points_.clear();
  points_.reserve(size_t(hRes_) * size_t(vRes_));

  const Scalar waterZ = Scalar(0.0);
  const Scalar eps    = Scalar(0.03);

  // ---- un lock global para leer ranges_ estable ----
  std::lock_guard<std::mutex> lk(rangesMtx_);

  unsigned int j0 = 0;
  for(unsigned int s=0;s<segs_;++s)
  {
    const unsigned int w = segW_[s];
    const auto& rr = ranges_[s];

    for(unsigned int i=0;i<vRes_;++i)
    {
      for(unsigned int u=0;u<w;++u)
      {
        const unsigned int j = j0 + u;
        const size_t idxG = size_t(i)*size_t(hRes_) + size_t(j);
        const size_t idxL = size_t(i)*size_t(w) + size_t(u);

        const Scalar r = Scalar(rr[idxL]);
        if(r <= Scalar(0.0) || r >= rangeMax_ * Scalar(0.9999)) continue;
        if(r < rangeMin_ || r > rangeMax_) continue;

        const Vector3 dirL = dirLUT_[idxG];
        const Vector3 hitW = originW_ + (basisW_ * dirL) * r;

        if(btFabs(hitW.z() - waterZ) < eps) continue;

        const Vector3 localP = basisWT_ * (hitW - scanTf_.getOrigin());

        LidarPoint p;
        p.x = (float)localP.x();
        p.y = (float)localP.y();
        p.z = (float)localP.z();
        p.intensity = (float)(1.0 - (r / rangeMax_));
        points_.push_back(p);
      }
    }

    j0 += w;
  }
}

void LidarGPU::InternalUpdate(Scalar dt)
{
  if(!gpuInit_) initGpu_();
  if(!gpuInit_) return;

  acc_ += dt;
  const Scalar period = (freqHz_ > Scalar(0.0)) ? (Scalar(1.0)/freqHz_) : Scalar(0.0);

  if(period > Scalar(0.0) && acc_ >= period && !scanRequested_)
  {
    acc_ -= period;
    requestScan_();
  }

  if(!scanRequested_) return;

  const uint32_t mask = readyMask_.load(std::memory_order_acquire);
  if(mask != allMask_) return;

  scanRequested_ = false;

  buildPointCloud_();
  if(callback_) callback_(points_);
}

std::vector<Renderable> LidarGPU::Render()
{
  std::vector<Renderable> items = LinkSensor::Render();

  if (isRenderable() && debugRender_)
  {
    Renderable item;
    item.type  = RenderableType::SENSOR_LINES;
    item.model = glMatrixFromTransform(getSensorFrame());
    item.data  = std::make_shared<std::vector<glm::vec3>>();

    auto pts = item.getDataAsPoints();
    {
      std::lock_guard<std::mutex> lk(debugMutex_);
      pts->reserve(pts->size() + debugLines_.size());
      for(const auto& p : debugLines_)
        pts->push_back(glm::vec3((float)p.x(), (float)p.y(), (float)p.z()));
    }
    items.push_back(item);
  }
  return items;
}

} // namespace sf
