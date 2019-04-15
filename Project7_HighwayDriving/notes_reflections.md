## Higway Driving Project - Notes & Reflections
Author: Tomas Mawyin
 
#### The objective of this project is generate a path planning algorithm to control the vehicle in a highway scenario. The goal is to drive smoothly without sudden jerk motions but trying to keep the speed limit.

---

#### 1. Straight Line Example

As explained in the project notes, the first thing I tested was following a straight path. This allowed me to understand how the car calculates its speed and how to generate a list of points. To achieve this, the code below generates points 0.5m appart, using the 0.02 second time step from the simulator, we drive at a speed of 25m/s or near 56mph.

```cpp
double dist_inc = 0.5;
          for (int i = 0; i < 50; ++i) {
            next_x_vals.push_back(car_x+(dist_inc*i)*cos(deg2rad(car_yaw)));
            next_y_vals.push_back(car_y+(dist_inc*i)*sin(deg2rad(car_yaw)));
          }
```

It was pointed out that a quick change in coordinate system, more specifically to the frenet coordinate system will allow the vehicle to keep it's lane. These points are generated using the helper functions provided with the project. There are two main things to consider for the next step; first, the velocity needs to change to avoid collisions, and second, the path generation needs to avoid the discotinuities that make the vehicle accelerate quicly.

#### 2. Smoothing Path

I used the video explanation to implement the path smoothing using the spline class. To perform this, we generated a list of points that are separated around 25 meters apart to build the spline. We take advantage of the last few points in the previous path to ensure continuity of the points being generated. The points being generated depend on the lane we want to be in which will allow us to move lanes later in the code. Below is an example of the points being generated:

```cpp
vector<double> next_wp0 = getXY(car_s + 25,  2 + 4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
vector<double> next_wp1 = getXY(car_s + 50,  2 + 4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
vector<double> next_wp2 = getXY(car_s + 75,  2 + 4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
vector<double> next_wp3 = getXY(car_s + 100, 2 + 4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
``` 

Like mentioned in the step above, we changed coordinate system to deal only in frenet coordinates, so the snippet of code above transforms to X-Y coordinate system to make sure the spline is in the right frame of reference. We now have points far enough to generate a spline, but we still need to take points along that spline to generate the car path. To do this, we linearize the spline by creating "N" points given a target distance from the car, then we use the spline add those spline points to the path.

```cpp
double N = target_dist/(0.02*ref_vel/2.24);
double x_point = x_add_on + target_x/N;
double y_point = s(x_point); 
```
We now have a smooth path that the car can follow. I decided to work with more points in the spline and include a larger path in the path planner (60 points) to see how the vehicle behaves. It turns out that 60 points didn't cause any issues in the final result so I decided to keep this configuration. Note that we haven't considered changing lanes or even working with other vehicles around us. 

#### 3. Including Sensor Fusion

In this section we will use Sensor Fusion from the simulator to deal with other vehicles in the road. We need to understand what lane each of the other vehicles in the road are and we can determine this by looking at the "d" coordinate value. We loop through the sensor fusion to check the data from the vehicles on the right side of the road. We use this information to determine other vehicles' speed and project where they would be during our path. The following code will help with this:

```cpp
for ( int i = 0; i < sensor_fusion.size(); i++ ) 
{
	// Let's find other car's speeds - we will need this for knowing if we might hit them
	double otherCar_vx = sensor_fusion[i][3];
	double otherCar_vy = sensor_fusion[i][4];
	double otherCar_speed = sqrt(otherCar_vx*otherCar_vx + otherCar_vy*otherCar_vy);

	// Finding other cars' "s" coordinate
	double otherCar_s = sensor_fusion[i][5];
	// Estimate car's position if using previous points. Projecting points to other car speeds
	otherCar_s += ((double)prev_size*0.02*otherCar_speed);

	// Gathering other vehicles' "d" coordinate value
	float d = sensor_fusion[i][6];

	// Let's check what lane the other vehicle is at. Note that this is possible since
	// sensor fusion only measures cars on our side of the road
	int otherCar_lane = floor(d/4.0);
	...
}
```

Once we determine what lane other cars are in, we can make a decision on what to do. We can either change lanes if there are no cars on the adjacent lane or reduce speed if the lanes next to our car are occupied. The decision to do this will be explained on the next section. The Q&A video was a great example on how to work with the sensor fusion and use it to project car's positions based on our path.

#### 4. Vehicle Behaviour

The final step was to control the vehicle behaviour by ajusting it's lane or velocity. Now that we know where the other cars are and which lanes they are using, we can see how close they are to our vehicle by setting up a gap distance. In my case, I used 30 meters and I checked the "s" coordinates to see if we are within that distance prior to making a decision.

I also use a few flags to check if I have a car to the left or to the right. In general, if there is a car in front of us that is within the 30 meters, we try to make a choice of chaning lanes by looking around to other vehicles, this includes looking back as well since we don't want to cut any vehicle off. If this is not possible, we slow down by using an acceleration decrease of 0.225 m/s2 to avoid any jerk motions. The vehicle also speeds up to try to maintain speed limits if there is no cars in front.

#### 5. Conclusions & Other Improvements

In general, we met the goals of this project:
- The vehicle never exceeds the speed limit and when possible goes at a maximum speed of 49.5 mph
- The vehicle drove more than 5 miles without any incidents
- Maximum acceleration and jerk were not exceeded
- No collisions were reported
- The vehicle stays within lane except when overtaking other vehicles

I played around with some of the parameters I used, like increasing the acceleration metric, increasing and decreasing the gap distance between the vehicles and using more points on the path planner but the setting provided above worked best for me. I would definitely enjoyed working on this project, and I must say that the video provided is a very great help to connect the dots between the multiple lessons leading to the project. I would also like to go back to the lessons and refine my knowledge to try some of the other approaches for jerk minimization, and implementing other cost functions.

