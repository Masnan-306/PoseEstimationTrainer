import cv2
import numpy as np
import torch
from monodepth2 import networks
from PoseModule import PoseDetector
# from cvzone.PoseModule import PoseDetector

from pdb import set_trace as bp

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16

# Load Monodepth2 model (example, you need to download and configure the model)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
encoder_path = 'mono_resnet50_640x192/encoder.pth'
decoder_path = 'mono_resnet50_640x192/depth.pth'

encoder = networks.ResnetEncoder(num_layers=50, pretrained=False)
loaded_dict_enc = torch.load(encoder_path, map_location=device)

filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()

depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
loaded_dict = torch.load(decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)
depth_decoder.to(device)
depth_decoder.eval()

def in_image(points, m):
    h, w = m.shape[0], m.shape[1]
    for x, y in points:
        if abs(x) >= w or abs(y) >= h:
            return False
    return True

def pointResize(original_size, new_size, point):
    original_height, original_width = original_size
    new_height, new_width = new_size
    x, y = point

    converted_x = int(x * (new_width / original_width))
    converted_y = int(y * (new_height / original_height))

    return converted_x, converted_y

# Function to calculate 3D angle of the right arm
def calculate_3d_angle(shoulder_2d, elbow_2d, wrist_2d, depth_map):
    if not in_image([shoulder_2d, elbow_2d, wrist_2d], depth_map):
        return 0
    
    # Retrieve depth values for each joint from the depth map
    shoulder_depth = depth_map[int(shoulder_2d[1]), int(shoulder_2d[0])]
    elbow_depth = depth_map[int(elbow_2d[1]), int(elbow_2d[0])]
    wrist_depth = depth_map[int(wrist_2d[1]), int(wrist_2d[0])]

    # Calculate 3D coordinates of joints
    shoulder_3d = np.array([shoulder_2d[0], shoulder_2d[1], shoulder_depth])
    elbow_3d = np.array([elbow_2d[0], elbow_2d[1], elbow_depth])
    wrist_3d = np.array([wrist_2d[0], wrist_2d[1], wrist_depth])

    # Calculate 3D vectors representing arm segments
    vector_shoulder_elbow = elbow_3d - shoulder_3d
    vector_elbow_wrist = wrist_3d - elbow_3d

    # Calculate angle using dot product
    angle_radians = np.arccos(np.dot(vector_shoulder_elbow, vector_elbow_wrist) /
                              (np.linalg.norm(vector_shoulder_elbow) * np.linalg.norm(vector_elbow_wrist)))

    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

# Open camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
detector = PoseDetector()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    input_frame = cv2.resize(frame, (640, 192))  # Resize to match Monodepth2 input size
    input_tensor = torch.from_numpy(input_frame.transpose((2, 0, 1))).float().unsqueeze(0)

    # Forward pass to obtain depth map
    with torch.no_grad():
        output = depth_decoder(encoder(input_tensor))

    depth_map = output[("disp", 0)].squeeze().cpu().numpy()

    frame = detector.findPose(frame, True)
    lmlist = detector.findPosition(frame)
    angle_3d = 0
    if lmlist:
        shoulder_2d = pointResize(frame.shape[:2], (192, 640), lmlist[RIGHT_SHOULDER][1:])
        elbow_2d =  pointResize(frame.shape[:2], (192, 640), lmlist[RIGHT_ELBOW][1:])
        wrist_2d = pointResize(frame.shape[:2], (192, 640), lmlist[RIGHT_WRIST][1:])
        
        # Calculate 3D angle using depth information
        angle_3d = calculate_3d_angle(shoulder_2d, elbow_2d, wrist_2d, depth_map)
        
    # Visualize depth map (optional)
    cv2.imshow("Depth Map", depth_map)

    # Display the angle
    cv2.putText(frame, f"3D Angle: {angle_3d:.2f} degrees", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Input Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
