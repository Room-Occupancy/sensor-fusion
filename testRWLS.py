import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as sp
import numpy as np
from operator import itemgetter
import time

def calculate_leverage(x, x_bar):
    n = len(x)
    return (1/n) + ((x - x_bar)**2) / np.sum((x - x_bar)**2)

def calculate_cooks_distance(y, y_pred, mse, p):
    return np.sum((y - y_pred)**2) / (p * mse)

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def weighted_least_squares(x, y, weights):
    # Ensure input lists have the same length
    if len(x) != len(y) or len(x) != len(weights):
        raise ValueError("Input lists must have the same length.")

    n = len(x)

    # Create design matrix X with a column of ones for the intercept
    X = np.column_stack((np.ones(n), x))

    # Create diagonal matrix W from the weights
    W = np.diag(weights)

    # Calculate the weighted least squares coefficients
    XtW = np.dot(X.T, W)
    XtWX_inv = np.linalg.inv(np.dot(XtW, X))
    coefficients = np.dot(np.dot(XtWX_inv, XtW), y)

    return coefficients

class SensorEstimator:
    
    def __init__(self, sensor_id, buffer_size=5, confidence_decay=5, confidence_scaling_factor=1, doWLS = False):
        self.sensor_id = sensor_id
        self.buffer_size = buffer_size
        self.confidence_decay = confidence_decay
        self.confidence_scaling_factor = confidence_scaling_factor
        self.buffer = []
        self.x = []
        self.weights = []
        self.y = []
        self.count = 1
        self.confidence = 0.0001
        self.forgettingweights = np.linspace(1, 0, self.buffer_size)
        self.temp = 0
        self.coefficients = (1, 1)
        self.time = 1
        self.doWLS = doWLS
    
    def setTime(self, newTime):
        self.time = newTime

    def update(self, new_count, new_confidence):
        # Update buffer
        if(new_count > self.count and len(self.buffer) > 0):
            new_confidence = 5 * new_confidence
            #new_confidence = min(new_confidence, 1)
        
        self.x.append(self.time % self.buffer_size)
        self.y.append(new_count)
        self.weights.append(new_confidence)

        if(len(self.x) >= self.buffer_size and self.doWLS):
            self.x.pop(0)
            self.y.pop(0)
            self.weights.pop(0)

            n = len(self.x)

            # Create design matrix X with a column of ones for the intercept
            X = np.column_stack((np.ones(n), self.x))

            # Create diagonal matrix W from the weights
            W = np.diag(self.weights)

            # Calculate the weighted least squares coefficients
            XtW = np.dot(X.T, W)
            XtWX_inv = np.linalg.inv(np.dot(XtW, X))
            self.coefficients = (np.dot(np.dot(XtWX_inv, XtW), self.y))
        
        if(new_count > self.count and len(self.buffer) > 0):
            #new_confidence = 2 * new_confidence
            new_confidence = min(new_confidence, 1)
        else:
            new_confidence = new_confidence * (1 - (abs(self.count - new_count)) / (self.count))
        
        self.buffer.append((new_count, new_confidence))
        

        if len(self.buffer) > self.buffer_size:
            #self.buffer = self.buffer * self.forgettingweights
            self.buffer.pop(0)



        # Filter out less confident values
        filtered_buffer = [(count, confidence) for count, confidence in self.buffer if confidence >= 1e-8]

        if not filtered_buffer:
            return  # Skip update if there are no confident values in the buffer

        # Calculate the rolling weighted average based on the filtered buffer data
        weighted_sum_counts = sum(count * confidence for count, confidence in filtered_buffer) 
        sum_weights = sum(confidence  for _, confidence in filtered_buffer)

        # Update the count as a rolling weighted average
        self.count = weighted_sum_counts / (sum_weights + 1e-8)

        # Update confidence based on the proximity of predicted and received values
        proximity_factor = 1 - (abs(self.count - new_count)) / (self.count)
        #self.confidence = ((self.confidence * self.confidence_decay) + (proximity_factor * new_confidence) * self.confidence_scaling_factor) / (self.confidence_decay)  # Updated confidence calculation
        self.confidence = sum_weights / self.buffer_size

        # Print intermediate results
        #print(f"Sensor {self.sensor_id}: Predicted Count: {self.count:.2f}, Confidence: {self.confidence:.2f}")
        #print("Updated Combined Count:", self.count)
        #print("----------------------")


# Initialize sensors with buffer size
sensor1 = SensorEstimator(sensor_id=1, buffer_size=200)
sensor2 = SensorEstimator(sensor_id=2, buffer_size=200)
sensor3 = SensorEstimator(sensor_id=3, buffer_size=200)
output = SensorEstimator(sensor_id=69, buffer_size=50)
outputSensor = SensorEstimator(sensor_id=420, buffer_size=50, doWLS=True)

sensors = [sensor1, sensor2, sensor3]

# Sample inputs for demonstration
n_obs = 10000
lowstart = 5
highstart= 15

target_occ = np.zeros(n_obs)
sensor1data = np.zeros(n_obs)
sensor2data = np.zeros(n_obs)
sensor3data = np.zeros(n_obs)

temp = np.random.randint(lowstart, highstart)
for i in range(n_obs):
    target_occ[i] = temp

    #generate a datapoint based on a skewed normal distribution located at the current "true count"
    sensor1data[i] = sp.skewnorm.rvs(-50, scale=3, loc=target_occ[i] + 1, size=1)
    sensor2data[i] = sp.skewnorm.rvs(-50, scale=4, loc=target_occ[i] + 1, size=1)
    sensor3data[i] = sp.skewnorm.rvs(-20, scale=5, loc=target_occ[i] + 1, size=1)
    
    #randomly increase count (someone enters room)
    if(np.random.randint(1, 150) == 1 and i < 0.7*n_obs):
        temp = temp + 1
    #randomly decrease count (someone leaves the room)
    if(np.random.randint(1, 75) == 0 and i < 0.7*n_obs):
        temp = temp - 1


np.around(sensor1data, 0)
np.around(sensor2data, 0)
np.around(sensor3data, 0)
weights_sensor1 = np.zeros(n_obs)
weights_sensor2 = np.zeros(n_obs)
weights_sensor3 = np.zeros(n_obs)

weight_exponent = 5

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

output_estimates = np.zeros(n_obs)

outputWLS = []
time = 0
for i in range(len(sensor1data)):
    sensor1.update(sensor1data[i], weights_sensor1[i])
    sensor2.update(sensor2data[i], weights_sensor2[i])
    sensor3.update(sensor3data[i], weights_sensor3[i])

    outputSensor.update(sensor1data[i], weights_sensor1[i])
    time = time + 1
    outputSensor.setTime(time)
    outputSensor.update(sensor2data[i], weights_sensor2[i])
    time = time + 1
    outputSensor.setTime(time)
    outputSensor.update(sensor3data[i], weights_sensor3[i])
    time = time + 1
    outputSensor.setTime(time)

    final_combined_count = (
        sensor1.count * sensor1.confidence
        + sensor2.count * sensor2.confidence
        + sensor3.count * sensor3.confidence
    ) / (sensor1.confidence + sensor2.confidence + sensor3.confidence)
    output.update(final_combined_count, ((sensor1.confidence + sensor2.confidence + sensor3.confidence)) / 3)
    output_estimates[i] = np.round(output.count, 0)
    outputWLS.append(np.round(outputSensor.coefficients[0], 1))

print("Target: ", target_occ)
print("WLS :", np.mean(outputWLS[n_obs-10:n_obs-1]))
print("Final Fused Count:", output_estimates[n_obs-1])
print("Fused std. dev: ", np.std(output_estimates))
print("Sensor 1 prediction:", np.mean(sensor1data))
print("Sensor 1 std. dev: ", np.std(sensor1data))
print("Sensor 2 prediction:", np.mean(sensor2data))
print("Sensor 2 std. dev: ", np.std(sensor2data))
print("Sensor 3 prediction:", np.mean(sensor3data))
print("Sensor 3 std. dev: ", np.std(sensor3data))

plt.plot(outputWLS)
plt.plot(target_occ)
plt.show()

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