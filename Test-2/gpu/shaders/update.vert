// Transform Feedback physics update vertex shader.
// Updates particle position and velocity each frame on the GPU.
// Runs orbital vortex, curl noise, attractor forces, damping, and bounding wrap.

#version 430

in vec3  in_position;
in vec3  in_velocity;
in float in_age;

out vec3  out_position;
out vec3  out_velocity;
out float out_age;

uniform vec3  attractors[6];
uniform float attractor_active;
uniform float gesture_mode;     // 0=attract, 1=repel, 2=vortex
uniform float dt;
uniform float time;
uniform float bounding_radius;

// Pseudo-random scalar from a float seed
float hash(float n) {
    return fract(sin(n) * 43758.5453123);
}

// Pseudo-random 3D vector from a 3D position
vec3 hash3(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

// Simple divergence-free curl noise for organic cloud drift
vec3 curl_noise(vec3 p) {
    float eps = 0.01;
    vec3 dx = vec3(eps, 0.0, 0.0);
    vec3 dy = vec3(0.0, eps, 0.0);
    vec3 dz = vec3(0.0, 0.0, eps);
    float x1 = hash3(p + dy).z - hash3(p - dy).z;
    float x2 = hash3(p + dz).y - hash3(p - dz).y;
    float y1 = hash3(p + dz).x - hash3(p - dz).x;
    float y2 = hash3(p + dx).z - hash3(p - dx).z;
    float z1 = hash3(p + dx).y - hash3(p - dx).y;
    float z2 = hash3(p + dy).x - hash3(p - dy).x;
    return vec3(x1 - x2, y1 - y2, z1 - z2) / (2.0 * eps);
}

void main() {
    vec3  pos = in_position;
    vec3  vel = in_velocity;
    float age = in_age + dt;

    // Attractor forces
    if (attractor_active > 0.5) {
        for (int i = 0; i < 6; i++) {
            vec3  delta = attractors[i] - pos;
            float dist  = length(delta) + 0.001;
            vec3  dir   = delta / dist;

            // Radial: attract (modes 0,2) or repel (mode 1)
            float sign = (gesture_mode > 0.5 && gesture_mode < 1.5) ? -1.0 : 1.0;
            vel += dir * sign * 0.18 / (dist * dist + 0.02) * dt;

            // Orbital tangent force creates the vortex spin
            vec3  up      = vec3(0.0, 1.0, 0.01);
            vec3  tangent = normalize(cross(dir, up));
            float orbit   = (gesture_mode > 1.5) ? 0.55 : 0.22;
            vel += tangent * orbit / (dist + 0.08) * dt;
        }
    }

    // Curl noise — organic drift (always active)
    vec3 npos = pos * 1.2 + vec3(time * 0.04, time * 0.03, time * 0.02);
    vel += curl_noise(npos) * 0.012 * dt;

    // Velocity damping
    float damp = (attractor_active > 0.5) ? 0.985 : 0.972;
    vel *= pow(damp, dt * 60.0);

    // Speed cap
    float spd = length(vel);
    if (spd > 2.5) vel = (vel / spd) * 2.5;

    pos += vel * dt;

    // Bounding sphere soft wrap-around
    if (length(pos) > bounding_radius) {
        float seed  = hash(float(gl_VertexID) + time * 0.01);
        float theta = seed * 6.283185;
        float phi   = acos(2.0 * hash(seed + 1.0) - 1.0);
        float r     = bounding_radius * (0.2 + 0.8 * hash(seed + 2.0));
        pos = vec3(r * sin(phi) * cos(theta),
                   r * sin(phi) * sin(theta),
                   r * cos(phi));
        vel *= 0.1;
    }

    out_position = pos;
    out_velocity = vel;
    out_age      = age;
}
