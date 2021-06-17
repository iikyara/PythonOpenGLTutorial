#version 460 core
uniform mat4 modelview;
uniform mat4 projection;
uniform mat3 normalMatrix;
uniform int have_position;
uniform int have_normals;
uniform int have_colors;
uniform int have_texcoords;
in vec4 position;
in vec3 normal;
in vec4 color;
in vec2 texcoord;
out vec4 P;
out vec3 N;
out vec4 outColor;

void main(void){
    outColor = color;
    P = modelview * position;
    N = normalize(normalMatrix * normal);
    gl_Position = projection * P;
}
