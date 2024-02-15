import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as sp


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)



class SensorEstimator:
    def __init__(self, sensor_id, buffer_size=5, confidence_decay=10, confidence_scaling_factor=10):
        self.sensor_id = sensor_id
        self.buffer_size = buffer_size
        self.confidence_decay = confidence_decay
        self.confidence_scaling_factor = confidence_scaling_factor
        self.buffer = []
        self.count = 0.0
        self.confidence = 0.0001

    def update(self, new_count, new_confidence):
        # Update buffer
        self.buffer.append((new_count, new_confidence))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # Filter out less confident values
        filtered_buffer = [(count, confidence) for count, confidence in self.buffer if confidence >= 0.01]

        if not filtered_buffer:
            return  # Skip update if there are no confident values in the buffer

        # Calculate the rolling weighted average based on the filtered buffer data
        weighted_sum_counts = sum(count * confidence for count, confidence in filtered_buffer)
        sum_weights = sum(confidence for _, confidence in filtered_buffer)

        # Update the count as a rolling weighted average
        self.count = weighted_sum_counts / (sum_weights + 1e-8)

        # Update confidence based on the proximity of predicted and received values
        proximity_factor = 1 - (abs(self.count - new_count)) / (self.count)
        self.confidence = ((self.confidence * self.confidence_decay) + (proximity_factor * new_confidence) * self.confidence_scaling_factor) / (self.confidence_decay)  # Updated confidence calculation

        # Print intermediate results
        print(f"Sensor {self.sensor_id}: Predicted Count: {self.count:.2f}, Confidence: {self.confidence:.2f}")
        print("Updated Combined Count:", self.count)
        print("----------------------")


# Initialize sensors with buffer size
sensor1 = SensorEstimator(sensor_id=1, buffer_size=200)
sensor2 = SensorEstimator(sensor_id=2, buffer_size=200)
sensor3 = SensorEstimator(sensor_id=3, buffer_size=200)

sensors = [sensor1, sensor2, sensor3]

# Sample inputs for demonstration
n_obs = 10000
lowstart = 40
highstart= 50

target_occ = np.zeros(n_obs)
sensor1data = np.zeros(n_obs)
sensor2data = np.zeros(n_obs)
sensor3data = np.zeros(n_obs)

temp = np.random.randint(lowstart, highstart)
for i in range(n_obs):
    target_occ[i] = temp

    #generate a datapoint based on a skewed normal distribution located at the current "true count"
    sensor1data[i] = sp.skewnorm.rvs(-50, scale=3, loc=target_occ[i] + 1, size=1)
    sensor2data[i] = sp.skewnorm.rvs(-50, scale=4, loc=target_occ[i] + 3, size=1)
    sensor3data[i] = sp.skewnorm.rvs(-20, scale=1, loc=target_occ[i] -15, size=1)
    
    #randomly increase count (someone enters room)
    if(np.random.randint(1, 50) == 0 and i < 0.7*n_obs):
        temp = temp + 1
    #randomly decrease count (someone leaves the room)
    if(np.random.randint(1, 75) == 0 and i < 0.7*n_obs):
        temp = temp - 1

#sensor1data = sp.skewnorm.rvs(-5, scale=3, loc=target_occ + 1, size=n_obs)
#sensor2data = sp.skewnorm.rvs(-10, scale=5, loc=target_occ + 1, size=n_obs)
#sensor3data = sp.skewnorm.rvs(-20, scale=8, loc=target_occ + 1, size=n_obs)

weights_sensor1 = np.zeros(n_obs)
weights_sensor2 = np.zeros(n_obs)
weights_sensor3 = np.zeros(n_obs)

weight_exponent = 20

for i in range(len(sensor1data)):
    weights_sensor1[i] = (
        clamp(1 - abs((sensor1data[i] - target_occ[i]) / target_occ[i]), 0, 1) ** weight_exponent
    )
    
    weights_sensor2[i] = (
        clamp(1 - abs((sensor2data[i] - target_occ[i]) / target_occ[i]), 0, 1) ** weight_exponent
    )
    weights_sensor3[i] = (
        clamp(1 - abs((sensor3data[i] - target_occ[i]) / target_occ[i]), 0, 1) ** weight_exponent
    )

streaming_data_sensor1 = zip(sensor1data, weights_sensor1)
streaming_data_sensor2 = zip(sensor2data, weights_sensor2)
streaming_data_sensor3 = zip(sensor3data, weights_sensor3)
output_estimates = np.zeros(n_obs)

for i in range(len(sensor1data)):
    sensor1.update(sensor1data[i], weights_sensor1[i])
    sensor2.update(sensor2data[i], weights_sensor2[i])
    sensor3.update(sensor3data[i], weights_sensor3[i])
    final_combined_count = (
        sensor1.count * sensor1.confidence
        + sensor2.count * sensor2.confidence
        + sensor3.count * sensor3.confidence
    ) / (sensor1.confidence + sensor2.confidence + sensor3.confidence)
    output_estimates[i] = final_combined_count
    # print("Target: ", target_occ)
    # print("Final Fused Count:", final_combined_count)

# Get the final combined results
final_combined_count, final_combined_confidence = (
    sum(sensor.count * sensor.confidence for sensor in sensors) /
    sum(sensor.confidence for sensor in sensors) if sensors else 0.0,  # Confidence is now a rolling average
    sum(sensor.confidence for sensor in sensors) / len(sensors) if sensors else 0.0  # Confidence is now a rolling average
)
print("Target: ", target_occ)
print("Final Fused Count:", final_combined_count)
print("Fused std. dev: ", np.std(output_estimates))
print("Sensor 1 prediction:", np.mean(sensor1data))
print("Sensor 1 std. dev: ", np.std(sensor1data))
print("Sensor 2 prediction:", np.mean(sensor2data))
print("Sensor 2 std. dev: ", np.std(sensor2data))
print("Sensor 3 prediction:", np.mean(sensor3data))
print("Sensor 3 std. dev: ", np.std(sensor3data))

#print(weights_sensor1)
#print(sensor1data)

plt.hist(sensor1data, bins=50, histtype="step")
plt.hist(sensor2data, bins=50, histtype="step")
plt.hist(sensor3data, bins=50, histtype="step")
plt.hist(output_estimates, bins=50, histtype="step")
plt.hist(target_occ[0:n_obs//5], bins=50, histtype="step")
plt.ylabel("Number of Occurances")
plt.xlabel("Occupancy Estimate")
plt.show()

plt.plot(sensor3data, color='darkseagreen', linewidth=0.5)
plt.plot(sensor2data, color='lightblue', linewidth=0.5)
plt.plot(sensor1data, color='lightcoral', linewidth=0.5)
plt.plot(target_occ, color='cyan', linewidth=2)
plt.plot(output_estimates, color='darkblue', linewidth=2)
plt.ylabel("Occupancy Estimate")
plt.xlabel("\"Time\"")

plt.show()