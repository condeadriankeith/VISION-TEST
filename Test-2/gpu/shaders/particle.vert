// Particle render pass vertex shader.
// Maps particle world-space position to screen space.
// Computes per-particle color from distance to nearest attractor (spectrum gradient).

#version 430

in vec3  in_position;
in vec3  in_velocity;
in float in_age;

out vec3  v_color;
out float v_alpha;

uniform mat4  u_mvp;
uniform vec3  attractors[6];
uniform float attractor_active;
uniform float viewport_h;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);

    // Distance to nearest attractor
    float min_dist = (attractor_active > 0.5) ? 1e9 : length(in_position) * 0.5;
    if (attractor_active > 0.5) {
        for (int i = 0; i < 6; i++) {
            float d = length(attractors[i] - in_position);
            if (d < min_dist) min_dist = d;
        }
    }

    // t=0 (very near) to t=1 (far) — drives color gradient
    float t = clamp(min_dist / 0.55, 0.0, 1.0);

    // 3-stop color spectrum: solar gold -> vibrant electric cyan -> celestial blue
    vec3 c_near = vec3(1.00, 0.95, 0.65);
    vec3 c_mid  = vec3(0.00, 0.92, 1.00);
    vec3 c_far  = vec3(0.20, 0.50, 1.00);

    vec3 color = (t < 0.5)
        ? mix(c_near, c_mid, t * 2.0)
        : mix(c_mid,  c_far, (t - 0.5) * 2.0);

    // Speed-boost: fast particles glow brighter (streak effect)
    float speed = length(in_velocity);
    color *= (1.0 + clamp(speed * 0.8, 0.0, 1.5));

    v_color = color;

    // Alpha: ensure a visible baseline opacity
    float age_fade  = clamp(in_age * 2.0, 0.35, 1.0);
    float dist_fade = 1.0 - clamp((length(in_position) - 0.8) / 1.5, 0.0, 0.65);
    v_alpha = age_fade * dist_fade * 0.95;

    // Point size: prominent, visible, depth-attenuated, larger near attractors
    float depth_w   = max(gl_Position.w, 0.5);
    float base_size = clamp(viewport_h * 0.009, 5.0, 12.0);
    float near_bonus = clamp((1.0 - t) * 8.0, 0.0, 8.0);
    gl_PointSize = (base_size + near_bonus) / max(depth_w * 0.28, 0.4);
}
