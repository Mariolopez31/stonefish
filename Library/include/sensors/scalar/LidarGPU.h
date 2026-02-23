#ifndef LIDAR_GPU_H
#define LIDAR_GPU_H


#include "sensors/scalar/Lidar.h"
#include "sensors/scalar/LinkSensor.h"
#include <vector>
#include <functional>
#include <mutex>
#include <atomic>
#include <memory>

namespace sf
{
  struct Renderable;

  class OpenGLDepthCamera;
  class Camera;

  using LidarCallback = std::function<void(const std::vector<LidarPoint>&)>;

  class LidarGPU : public LinkSensor
  {
  public:
    LidarGPU(std::string name,
             unsigned int horizontalRes,
             unsigned int verticalRes,
             Scalar horizontalFovDeg,
             Scalar verticalFovDeg,
             Scalar minRange,
             Scalar maxRange,
             Scalar frequency);

    ~LidarGPU() override;

    SensorType getType() const override;
    ScalarSensorType getScalarSensorType() const override;

    void InternalUpdate(Scalar dt) override;
    void setResponseCallback(LidarCallback cb);

    std::vector<Renderable> Render() override;

  private:
    // --- specs ---
    unsigned int hRes_, vRes_;
    Scalar fovHdeg_, fovVdeg_;
    Scalar rangeMin_, rangeMax_;
    Scalar freqHz_;

    // tu recorte “arriba”
    Scalar vMinDeg_ = Scalar(0.5);
    Scalar vMaxDeg_ = Scalar(52.0);

    // --- output ---
    std::vector<LidarPoint> points_;
    LidarCallback callback_ = nullptr;

    // debug frustum (igual que antes)
    bool debugRender_ = true;
    std::vector<Vector3> debugLines_;
    std::mutex debugMutex_;
    void buildDebug_();

    // --- GPU rig: 4 segmentos de 90° ---
    bool gpuInit_ = false;
    bool scanRequested_ = false;

    unsigned int segs_ = 4;
    std::vector<unsigned int> segW_;                 // widths por seg
    std::vector<OpenGLDepthCamera*> views_;          // las views viven en OpenGLContent
    std::vector<std::vector<float>> ranges_;         // [seg][w*vRes]
    std::mutex rangesMtx_;
    std::atomic<uint32_t> readyMask_{0};
    uint32_t allMask_ = 0;

    Transform scanTf_;
    Vector3   originW_;
    btMatrix3x3 basisW_;
    btMatrix3x3 basisWT_;

    // LUT dirs (hRes*vRes) en frame del sensor
    std::vector<Vector3> dirLUT_;

    // timing
    Scalar acc_ = Scalar(0.0);

    // receptor para cada view
    struct GpuReceiver;
    std::vector<std::unique_ptr<GpuReceiver>> rx_;

    void initGpu_();
    void buildDirLUT_();
    void requestScan_();
    void onRanges_(unsigned int seg, const float* src);

    void buildPointCloud_();
  };
}

#endif
