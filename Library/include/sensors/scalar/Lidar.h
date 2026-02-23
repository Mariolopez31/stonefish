// Library/include/sensors/scalar/Lidar.h
#pragma once

#include "sensors/scalar/LinkSensor.h"
#include <vector>
#include <functional>
#include <mutex>

class btDynamicsWorld;
class btCollisionObject;

namespace sf
{

struct LidarPoint { float x, y, z, intensity; };

class Lidar : public LinkSensor
{
public:
  using LidarCallback = std::function<void(const std::vector<LidarPoint>&)>;

  Lidar(std::string name,
        unsigned int horizontalRes,
        unsigned int verticalRes,
        Scalar fovH,
        Scalar fovV,
        Scalar minRange,
        Scalar maxRange,
        Scalar frequency);

  ~Lidar() override;

  SensorType getType() const override;
  ScalarSensorType getScalarSensorType() const override;

  void setResponseCallback(LidarCallback cb);

  std::vector<Renderable> Render() override;

protected:
  void InternalUpdate(Scalar dt) override;

private:
  // LUT
  void buildRayLUT_();
  void buildDebugLUT_();

  // skip/self/water
  bool isWaterCandidate_(const btCollisionObject* obj, Scalar waterZ) const;
  void rebuildSkipList_(btDynamicsWorld* world, const Vector3& sensorPos);
  bool isSkipHit_(const btCollisionObject* obj) const;

private:
  unsigned int hRes = 0;
  unsigned int vRes = 0;
  Scalar fovH = Scalar(0);
  Scalar fovV = Scalar(0);
  Scalar rangeMin = Scalar(0);
  Scalar rangeMax = Scalar(0);
  Scalar freqHz_  = Scalar(0);

  // scan scheduling
  unsigned int scanCursor_ = 0;
  double raysAcc_ = 0.0;

  // frozen scan frame
  Transform scanTf_;
  Matrix3   scanBasis_;
  Matrix3   scanBasisT_;
  Vector3   scanOrigin_;

  // output
  std::vector<LidarPoint> bufferPoints;
  LidarCallback callback_ = nullptr;

  // LUT dirs (local)
  std::vector<Vector3> rayDirsLocal_;

  // debug
  bool debugRender = false; // si ya lo tenías en LinkSensor, quita esto
  std::mutex debugMutex;
  std::vector<Vector3> debugLinesWrite;
  std::vector<Vector3> debugLinesRender;

  // skip list
  std::vector<const btCollisionObject*> skip_;
  const btCollisionObject* waterObj_ = nullptr;
  unsigned int skipRebuildCounter_ = 0;
};

} // namespace sf
