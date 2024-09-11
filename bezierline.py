import numpy as np
import cv2

# Set initial size of the display window
win_width, win_height = 800, 800

# Set size of world-coordinate clipping window with padding
padding = 20
xwc_min, xwc_max = -240.0, 240.0
ywc_min, ywc_max = 240.0, -240.0  # Note: y-axis is inverted in OpenCV

def world_to_screen(x, y):
    screen_x = int((x - xwc_min) / (xwc_max - xwc_min) * win_width)
    screen_y = int((y - ywc_min) / (ywc_max - ywc_min) * win_height)
    return screen_x, screen_y

def plot_point(img, point, color=(0, 0, 255), size=1):
    x, y = world_to_screen(point[0], point[1])
    cv2.circle(img, (x, y), size, color, -1)

def binomial_coeffs(n):
    return [1] + [0] * n

def compute_bez_pt(u, ctrl_pts, C):
    n = len(ctrl_pts) - 1
    bez_pt = np.zeros(3)
    for k in range(len(ctrl_pts)):
        bez_blend_fcn = C[k] * (u ** k) * ((1 - u) ** (n - k))
        bez_pt += ctrl_pts[k] * bez_blend_fcn
    return bez_pt

def bezier(ctrl_pts, n_bez_curve_pts):
    img = np.ones((win_height, win_width, 3), dtype=np.uint8) * 255
    
    C = binomial_coeffs(len(ctrl_pts) - 1)
    for i in range(len(ctrl_pts)):
        C[i] = np.prod(range(len(ctrl_pts), i, -1)) // np.prod(range(1, len(ctrl_pts) - i + 1))
    
    # Plot Bezier curve
    for k in range(n_bez_curve_pts + 1):
        u = float(k) / float(n_bez_curve_pts)
        bez_curve_pt = compute_bez_pt(u, ctrl_pts, C)
        plot_point(img, bez_curve_pt, color=(0, 0, 255), size=1)
    
    # Plot control points and connecting lines
    for i, pt in enumerate(ctrl_pts):
        x, y = world_to_screen(pt[0], pt[1])
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        if i > 0:
            prev_x, prev_y = world_to_screen(ctrl_pts[i-1][0], ctrl_pts[i-1][1])
            cv2.line(img, (prev_x, prev_y), (x, y), (0, 255, 0), 1)
    
    return img

def main():
    n_ctrl_pts = 4
    n_bez_curve_pts = 2000
    ctrl_pts = np.array([
        [-40.0, -40.0, 0.0],
        [-10.0, 200.0, 0.0],
        [10.0, -200.0, 0.0],
        [40.0, 40.0, 0.0]
    ])

    img = bezier(ctrl_pts, n_bez_curve_pts)
    
    cv2.imshow("Bezier Curve", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()