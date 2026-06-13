/*
 * UAV Navigation Controller
 * Connect to Python viewer via TCP port 5599
 */

#include <webots/robot.h>
#include <webots/gps.h>
#include <webots/compass.h>
#include <webots/camera.h>
#include <webots/motor.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#define TIME_STEP 32

static WbDeviceTag gps, compass, camera;
static WbDeviceTag motors[4];
static int client_socket = -1;
static int server_socket = -1;

// Send data to Python viewer
void send_data(double x, double y, double z, double yaw) {
    if (client_socket < 0) return;

    char buffer[256];
    snprintf(buffer, sizeof(buffer),
             "{\"pose\": [%.3f, %.3f, %.3f, %.3f]}\n", x, y, z, yaw);

    send(client_socket, buffer, strlen(buffer), 0);
}

// Receive command from Python viewer
void receive_command() {
    if (client_socket < 0) return;

    char buffer[256];
    int n = recv(client_socket, buffer, sizeof(buffer) - 1, MSG_DONTWAIT);

    if (n > 0) {
        buffer[n] = '\0';

        // Parse simple commands: {"vx": 0.5} or {"yaw_rate": 0.1}
        double vx = 0, yaw_rate = 0;
        char *p;

        if ((p = strstr(buffer, "\"vx\":"))) {
            sscanf(p + 5, "%lf", &vx);
        }
        if ((p = strstr(buffer, "\"yaw_rate\":"))) {
            sscanf(p + 11, "%lf", &yaw_rate);
        }

        // Set motor velocities based on command
        // For simplicity, just set all propellers to same speed
        for (int i = 0; i < 4; i++) {
            wb_motor_set_velocity(motors[i], vx * 50.0 + yaw_rate * 10.0);
        }
    }
}

// Accept viewer connection
void accept_connection() {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);

    client_socket = accept(server_socket, (struct sockaddr *)&client_addr, &client_len);

    if (client_socket >= 0) {
        printf("Python viewer connected!\n");
    }
}

int main() {
    wb_robot_init();

    // Setup devices
    gps = wb_robot_get_device("gps");
    compass = wb_robot_get_device("compass");
    camera = wb_robot_get_device("camera");

    if (gps) wb_gps_enable(gps, TIME_STEP);
    if (compass) wb_compass_enable(compass, TIME_STEP);
    if (camera) wb_camera_enable(camera, TIME_STEP);

    // Get motors
    char motor_name[20];
    for (int i = 0; i < 4; i++) {
        sprintf(motor_name, "propeller%d", i);
        motors[i] = wb_robot_get_device(motor_name);
        if (motors[i]) {
            wb_motor_set_position(motors[i], INFINITY);
            wb_motor_set_velocity(motors[i], 0);
        }
    }

    // Create TCP server on port 5599
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in server_addr;
    int opt = 1;
    setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    server_addr.sin_port = htons(5599);

    bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr));
    listen(server_socket, 1);
    printf("Waiting for Python viewer on port 5599...\n");

    // Main loop
    while (wb_robot_step(TIME_STEP) != -1) {
        // Accept new connection if needed
        if (client_socket < 0) {
            accept_connection();
        }

        // Get pose
        double x = 0, y = 0, z = 0, yaw = 0;

        if (gps) {
            const double *pos = wb_gps_get_values(gps);
            x = pos[0];
            y = pos[1];
            z = pos[2];
        }

        if (compass) {
            const double *north = wb_compass_get_values(compass);
            yaw = atan2(north[0], north[2]);
        }

        // Send to viewer
        send_data(x, y, z, yaw);

        // Receive command from viewer
        receive_command();

        // Small delay to prevent busy loop
        usleep(1000);
    }

    // Cleanup
    if (client_socket >= 0) close(client_socket);
    if (server_socket >= 0) close(server_socket);

    wb_robot_cleanup();
    return 0;
}