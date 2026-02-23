#pragma once

#include "StonefishCommon.h"
#include "sensors/scalar/LinkSensor.h"

#include <functional>
#include <vector>

namespace sf {

struct LivoxPoint
{
  uint32_t offset_time;   // ns desde timebase
  float x, y, z;
  uint8_t reflectivity;
  uint8_t tag;
  uint8_t line;
};

class LivoxMid360CPU : public LinkSensor
{
public:
  using LivoxCallback = std::function<void(uint64_t timebase_ns,
                                           uint8_t lidar_id,
                                           const std::vector<LivoxPoint>& points)>;

  LivoxMid360CPU(std::string name,
                 unsigned int horizontalRes,
                 unsigned int verticalRes,
                 Scalar minRange,
                 Scalar maxRange,
                 Scalar frameRateHz);

  ~LivoxMid360CPU() override = default;

  SensorType getType() const override { return SensorType::LIDAR; }
  ScalarSensorType getScalarSensorType() const override { return ScalarSensorType::LIDAR; }

  void setResponseCallback(LivoxCallback cb) { callback_ = std::move(cb); }

  void setFovDeg(Scalar fovH_deg=Scalar(360.0), Scalar vMin_deg=Scalar(-7.0), Scalar vMax_deg=Scalar(52.0));
  void setPointRate(Scalar pts_per_sec);     // para offset_time
  void setLines(uint8_t lines=4);
  void setNonRepetitive(bool on=true);
  void setLidarId(uint8_t id);

protected:
  void InternalUpdate(Scalar dt) override;

private:
  void buildRayLUT_(Scalar yawOffsetDeg);
  void rebuildSkipList_(btDynamicsWorld* world, const Vector3& sensorPos);

  // config
  unsigned int hRes_{0}, vRes_{0};
  Scalar rangeMin_{0.1}, rangeMax_{70.0};
  Scalar frameHz_{10.0};
  Scalar pointRate_{200000.0};
  Scalar fovHdeg_{360.0};
  Scalar vMinDeg_{-7.0}, vMaxDeg_{52.0};
  uint8_t lines_{4};
  bool nonRepetitive_{true};
  uint8_t lidarId_{0};

  // scan
  std::vector<Vector3> rayDirsLocal_;     // size vRes*hRes
  std::vector<LivoxPoint> points_;
  unsigned int scanCursor_{0};
  double raysAcc_{0.0};
  uint64_t simTimeNs_{0};
  uint64_t scanTimebaseNs_{0};
  uint64_t frameIndex_{0};

  Transform scanTf_;
  Matrix3 scanBasis_;
  Matrix3 scanBasisT_;
  Vector3 scanOrigin_;

  std::vector<const btCollisionObject*> skip_;
  unsigned int skipRebuildCounter_{0};

  LivoxCallback callback_{nullptr};
  unsigned int hStep_{2};

  std::vector<uint32_t> rayIdxToGlobal_;
};

} // namespace sf