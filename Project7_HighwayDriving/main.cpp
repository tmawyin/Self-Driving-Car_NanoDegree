#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }


  // Need to define some variables:
  // lane - the lane that the car is in, 0 is the left lane, 1 the middle, and 2 is the right lane
  // Note that lane measures 4m so this will be adjusted later
  int lane = 1;

  // ref_vel (mph) - this is the velocity that the vehicle should drive at. We will start at 0mph to start
  double ref_vel = 0.0; 

  h.onMessage([&ref_vel, &lane, &map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
               &map_waypoints_dx,&map_waypoints_dy]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) 
  {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object

          // Main car's localization Data
            double car_x = j[1]["x"];
            double car_y = j[1]["y"];
            double car_s = j[1]["s"];
            double car_d = j[1]["d"];
            double car_yaw = j[1]["yaw"];
            double car_speed = j[1]["speed"];

            // Previous path data given to the Planner
            auto previous_path_x = j[1]["previous_path_x"];
            auto previous_path_y = j[1]["previous_path_y"];
            // Previous path's end s and d values
            double end_path_s = j[1]["end_path_s"];
            double end_path_d = j[1]["end_path_d"];

            // Sensor Fusion Data, a list of all other cars on the same side of the road.
            auto sensor_fusion = j[1]["sensor_fusion"];

            // Calculating the size of the previous path. 
            // This includes all points that have not been covered by the car at each iteration
            int prev_size = previous_path_x.size();

            // Moving our car "s" coordinate to the previous path "s" value.
            if (prev_size > 0) 
            {
              car_s = end_path_s;
            }

            // Variables below will help determine where cars are so that we can change lanes if possible
            bool car_ahead = false;
            bool car_left = false;
            bool car_righ = false;

            const double MAX_SPEED = 49.5; // This is the max speed we will allow (close to the speed limit in mph)
            const double MAX_ACC = .224;   // This is the max acceleration allowed
            const double MAX_GAP = 30.0;   // This it the max gap between vehicle in front of us

            // To avoind any cars in the road we need to go over the sensor fusion data
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
                
                // Now we can check if the car is in our lane or in a different lane and how far it is from us
                if ( otherCar_lane == lane ) 
                {
                  car_ahead |= (otherCar_s > car_s) && ((otherCar_s - car_s) < MAX_GAP);
                } 
                // Car on the left side
                else if ( otherCar_lane == lane-1 ) 
                {
                  car_left |= ((car_s - MAX_GAP) < otherCar_s) && ((car_s + MAX_GAP) > otherCar_s);
                } 
                // Car on the right side
                else if ( otherCar_lane == lane+1 ) 
                {
                  car_righ |= ((car_s - MAX_GAP) < otherCar_s) && ((car_s + MAX_GAP) > otherCar_s);
                }
            }

            // Now based on what cars we have near us, we can work on changing lanes or reduce speed
            double speed_diff = 0;

            // For a car ahead let's see if we can change lane
            if ( car_ahead ) 
            { 
              // No car is on the left lane and that side of the road is open
              if ( !car_left && lane > 0 ) 
              {
                lane--; // Change lane left.
              } 
              // No car is on the right lane and that side of the road is open
              else if ( !car_righ && lane != 2 )
              {
                lane++; // Change lane right.
              }
              // If no lanes are open, we reduce speed at a decent acceleration to avoid jerk 
              else 
              {
                speed_diff -= MAX_ACC;
              }
            } 
            else 
            {
              if ( lane != 1 ) 
              { // From a left or right lanes we can only move to the center lane
                if ( ( lane == 0 && !car_righ ) || ( lane == 2 && !car_left ) ) 
                {
                  lane = 1;
                }
              }
              // We can increase speed if we are below our maximum set speed (near speed limit)
              if ( ref_vel < MAX_SPEED ) 
              {
                speed_diff += MAX_ACC;
              }
            }

            // ptsx and ptsy are sparce points used to calculate the spline.
            // This will serve to smooth the path of the vehicle
            vector<double> ptsx;
            vector<double> ptsy;

            // The reference variables below holds either where the car is at or
            // at the previous path endpoint
            double ref_x = car_x;
            double ref_y = car_y;
            double ref_yaw = deg2rad(car_yaw);

            // Let's check how many points in the previous path we have left
            // If we are close to an empty set of points, we want to use points tangent to the car yaw
            if ( prev_size < 2 ) 
            {
                double prev_car_x = car_x - cos(car_yaw);
                double prev_car_y = car_y - sin(car_yaw);

                // We can now push these two last points to the spline calculation list
                ptsx.push_back(prev_car_x);
                ptsx.push_back(car_x);
                ptsy.push_back(prev_car_y);
                ptsy.push_back(car_y);
            } 
            // For multiple points, we pick the last two points in the path to generate that spline list 
            else 
            {
                // We pick the last point in the path
                ref_x = previous_path_x[prev_size - 1];
                ref_y = previous_path_y[prev_size - 1];

                // We pick the second last point in the path
                double ref_x_prev = previous_path_x[prev_size - 2];
                double ref_y_prev = previous_path_y[prev_size - 2];
                // We can calculate the yaw at end of the path points for reference
                ref_yaw = atan2(ref_y-ref_y_prev, ref_x-ref_x_prev);

                ptsx.push_back(ref_x_prev);
                ptsx.push_back(ref_x);
                ptsy.push_back(ref_y_prev);
                ptsy.push_back(ref_y);
            }

            // We can use frenet to space points far apart enough, these points will be part of the spline calculation
            vector<double> next_wp0 = getXY(car_s + 30,  2 + 4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
            vector<double> next_wp1 = getXY(car_s + 60,  2 + 4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
            vector<double> next_wp2 = getXY(car_s + 90,  2 + 4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
            // vector<double> next_wp3 = getXY(car_s + 100, 2 + 4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);

            ptsx.push_back(next_wp0[0]);
            ptsx.push_back(next_wp1[0]);
            ptsx.push_back(next_wp2[0]);
            // ptsx.push_back(next_wp3[0]);

            ptsy.push_back(next_wp0[1]);
            ptsy.push_back(next_wp1[1]);
            ptsy.push_back(next_wp2[1]);
            // ptsy.push_back(next_wp3[1]);

            // The ide is to translate all the spline points into local car coordinates to make calculations easier .
            for ( int i = 0; i < ptsx.size(); i++ ) 
            {
              double shift_x = ptsx[i] - ref_x;
              double shift_y = ptsy[i] - ref_y;

              ptsx[i] = shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw);
              ptsy[i] = shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw);
            }

            // Making use of the spline framework. The spline points are use to generate the spline and we use the
            // handle "s" as a function y = s(x).
            tk::spline s;
            s.set_points(ptsx, ptsy);

            // Here are the points the path planner will be using
            vector<double> next_x_vals;
            vector<double> next_y_vals;
            
            // Let's push all the previous points since those were already part of the path planning
            for (int i = 0; i < previous_path_x.size(); i++ ) 
            {
              next_x_vals.push_back(previous_path_x[i]);
              next_y_vals.push_back(previous_path_y[i]);
            }

            // target_x and target_y are points in the spline that will be used to linearize the path
            // and obtain the points along the spline to be included in the path planning
            double target_x = 25.0;
            double target_y = s(target_x);
            // target_dist is the straigh distance from the car to the target point
            double target_dist = sqrt(target_x*target_x + target_y*target_y);

            // We will use this variable to keep adding X-values to the passed to the spline function.
            double x_add_on = 0;

            // We can loop through th points that are missing in the path to build the path continuously (70 points total)
            for( int i = 1; i < (60 - previous_path_x.size()); i++ ) 
            {
              ref_vel += speed_diff;
              if ( ref_vel > MAX_SPEED ) 
              {
                ref_vel = MAX_SPEED;
              } 
              else if ( ref_vel < MAX_ACC ) 
              {
                ref_vel = MAX_ACC;
              }

              // Here is were we linearize the spline to space points along the spline
              double N = target_dist / (0.02 * ref_vel / 2.24);
              double x_point = x_add_on + (target_x / N);
              double y_point = s(x_point);

              x_add_on = x_point;

              double x_ref = x_point;
              double y_ref = y_point;

              // Translating back to the right coordinate system - from car to global coordinates
              // Note that we did a translation/rotation operation before, so here we do the opposite
              x_point = x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw);
              y_point = x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw);

              x_point += ref_x;
              y_point += ref_y;

              next_x_vals.push_back(x_point);
              next_y_vals.push_back(y_point);
            }

            json msgJson;

            msgJson["next_x"] = next_x_vals;
            msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}