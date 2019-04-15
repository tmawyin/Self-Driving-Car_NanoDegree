/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::default_random_engine;
using std::normal_distribution;
using std::numeric_limits;
using std::discrete_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  // We start with 100 particles to see how effective the filter is
  num_particles = 50;  // TODO: Set the number of particles

  // Setting the random_engine generator
  default_random_engine gen;

  // Generating a normal (Gaussian) distribution for x, y, and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Creating all particles and their weights set to 1
  for (int i=0; i< num_particles; i++)
  {
    // Creating a single particle and its properties
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
    weights.push_back(1);
  }

  // Setting the particles to intialized
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // Setting the random_engine generator
  default_random_engine gen;
  // Generating a normal (Gaussian) distribution for x, y, and theta [mean @ 0]
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);  

  // Let's go through each particle and generate a prediction using the
  // time step, velocity, and yaw_rate
  for (int i = 0; i< num_particles; i++)
  {
    // Making sure the yaw rate is not zero (avoid diving by 0)
    if (abs(yaw_rate) != 0) 
    {
      // Using equations of motion for position and heading
      particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
      particles[i].theta += yaw_rate * delta_t;
      
    } 
    // If there is no yaw rate, we can use the velocity component to update position
    else 
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta); 
    }

    // Add noise to the particles
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);

  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  // Iterate through all observations (asume all observations are in map-coordinates)
  for(auto &obs: observations)
  {
    // Inital minimum distance set to a maximum possible value
    double min_dist = numeric_limits<double>::max();

    // Initializing a temporary observation id to -1 to ensure that the mapping was found for observation
    int temp_id = -1;

    // Finding the nearest neighbor
    for(auto &pred: predicted)
    {
      double actual_dist = dist(pred.x, pred.y, obs.x, obs.y);

      // Update the observation id if a closer prediction is found
      if (actual_dist <= min_dist) 
      {
        min_dist = actual_dist;
        temp_id = pred.id;
      }
    }

    // Updating to the final observation id
    obs.id = temp_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  int particle_num = 0;
  // Iterating through all particles
  for (auto &p: particles)
  {
    // Transforming all observations to the map coordinate:
    // The following vector will hold all observations in the map coordinate
    vector<LandmarkObs> mapCord_obs(observations.size());
    
    // We need to loop through the observations
    for (unsigned int i = 0; i < observations.size(); i++)
    {
      // Applying transformation matrix
      mapCord_obs[i].x = observations[i].x*cos(p.theta) - observations[i].y*sin(p.theta) + p.x;
      mapCord_obs[i].y = observations[i].x*sin(p.theta) + observations[i].y*cos(p.theta) + p.y;
      // Keeping the same "id" as the observation
      mapCord_obs[i].id = observations[i].id;
    }

    // Gathering all the landmarks that are within the sensor range:
    vector<LandmarkObs> closeby_landmarks;

    // Iterating through the list of landmarks in the map
    for (auto const &lndmk: map_landmarks.landmark_list)
    {
      // Calculating particle to landmark distance
      double distance = dist(p.x, p.y, lndmk.x_f, lndmk.y_f);
      if (distance <= sensor_range)
      {
        // Saving all landmarks within range
        LandmarkObs tempLndmk;
        tempLndmk.id = lndmk.id_i;
        tempLndmk.x = lndmk.x_f;
        tempLndmk.y = lndmk.y_f;
        closeby_landmarks.push_back(tempLndmk);
      }
    }

    // We can do a data association to find nearest neighbour
    dataAssociation(closeby_landmarks, mapCord_obs);

    // Updating weights:
    double prob = 1.0;
    double normalizer = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
    
    for (unsigned int j=0; j < mapCord_obs.size(); j++)
    {
      // Getting the nearest landmarks
      LandmarkObs near_lndmk;
      near_lndmk.x = map_landmarks.landmark_list[ mapCord_obs[j].id - 1 ].x_f;
      near_lndmk.y = map_landmarks.landmark_list[ mapCord_obs[j].id - 1 ].y_f;

      // Calculating probability
      double gauss = exp(-1 * (pow((mapCord_obs[j].x-near_lndmk.x),2)/(2*pow(std_landmark[0],2)) + pow((mapCord_obs[j].y-near_lndmk.y),2)/(2*pow(std_landmark[1],2)) ));
      prob *= normalizer * gauss;
    }
    
    // Setting the particle weight
    p.weight = prob;
    weights[particle_num] = p.weight;
    particle_num += 1;

  } // particle loop end
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // Setting up the discrete distribution function
  default_random_engine gen;
  discrete_distribution<int> discrete_dist(weights.begin(), weights.end());

  // Generating a temporary resampling vector
  vector<Particle> resample_particles(particles.size());

  for(int i=0; i < num_particles; i++) 
  {
    resample_particles[i] = particles[discrete_dist(gen)];
  }

  particles = resample_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}