import math
import numpy
import base_keys
from DataFormat import datatypes_helper
from Processors.Yolov8.video_detection import VideoDetection as YoloDetector
from Utilities import image_utility
from Utilities import logging_utility
from base_component import BaseComponent

import cv2 as cv

from DataFormat.ProtoFiles.Template import template_data_pb2

from APIs.waddle import Waddle, get_news_nearby, EMAIL_SENDER

DATATYPE_PHOTO_TEMPLATE_IMAGE_FRAME_DATA = datatypes_helper.get_key_by_name("PHOTO_TEMPLATE_IMAGE_FRAME_DATA")
DATATYPE_TEMPLATE_DATA = datatypes_helper.get_key_by_name("TEMPLATE_DATA")


_logger = logging_utility.setup_logger(__name__)


class PhotoTemplateService(BaseComponent):
    """
    This service handles data processing, including object detection using YOLOv8 and generating data for a template
    scene. It processes data from both WebSocket and camera sources.
    """

    SUPPORTED_DATATYPES = {
        "PHOTO_TEMPLATE_IMAGE_FRAME_DATA",
        # "REQUEST_TEMPLATE_DATA"
    }

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.image_detector = YoloDetector()

    def run(self, raw_data: dict) -> None:
        super().set_component_status(base_keys.COMPONENT_IS_RUNNING_STATUS)

        origin = raw_data[base_keys.ORIGIN_KEY]

        if origin == base_keys.WEBSOCKET_WIDGET:
            datatype = raw_data[base_keys.WEBSOCKET_DATATYPE]
            message = raw_data[base_keys.WEBSOCKET_MESSAGE]
            data = raw_data[base_keys.WEBSOCKET_DATA]

            self._handle_websocket_data(datatype, message, data, raw_data)

    def _handle_websocket_data(self, socket_data_type, decoded_data, data, raw_data):
        if socket_data_type == DATATYPE_PHOTO_TEMPLATE_IMAGE_FRAME_DATA:
            self._handle_image_frame_data(data, raw_data)

    def _handle_image_frame_data(self, data, raw_data):
        # Convert frame from JPG to numpy array

        im = image_utility.get_frame_from_bytes(data.image[0])
        rgbim = cv.cvtColor(im, cv.COLOR_BGRA2RGB)

        INPUT_IMAGE_PATH = 'temp.jpg'
        cv.imwrite(INPUT_IMAGE_PATH, rgbim)

        params = {k: v for k, v in [tuple(kv.split(':')) for kv in data.params.split(';')]}
        latitude = float(params['latitude'])
        longitude = float(params['longitude'])
        
        IMAGE_PATH = "union_sq.jpg"
        VIDEO_PATH = 'video.mp4'
        EMAIL_RECEIVER = EMAIL_SENDER
        # EMAIL_RECEIVER = 'limjiehan02@gmail.com'

        vision_client = Waddle.VisionClient()
        location_info, sources = get_news_nearby(latitude, longitude)
        safe_or_danger_string, summary_text = vision_client.analyze_location_and_image(location_info, IMAGE_PATH)
        report_text = vision_client.create_report(location_info, summary_text, sources)
        audio_file = vision_client.generate_tts_summary(summary_text)
        vision_client.send_email(report_text, safe_or_danger_string, EMAIL_RECEIVER, audio_file, INPUT_IMAGE_PATH, VIDEO_PATH)

        # # rgbim = image_utility.rgb_image(im)
        # self.image_detector.run(rgbim)

        # # Get the detection results.
        # last_detection = self.image_detector.get_last_detection()

        # # As a demo, form a comma-separated string with class_label|score
        # detectionstr = ""
        # detection_count = len(last_detection["class_id"])

        # if detection_count > 0:
        #     detection_str_tokens = []
        #     class_labels = self.image_detector.get_class_labels()
        #     for i in range(detection_count):
        #         class_id = last_detection["class_id"][i]
        #         class_label = class_labels[class_id]
        #         conf = last_detection["confidence"][i]
        #         detection_str_tokens.append(f"{class_label}|{conf:0.2f}")
        #     detectionstr = ",".join(detection_str_tokens)
        
        template_data = self.build_template_data(safe_or_danger_string, None, "")

        # As the event is passed to 2 output handlers, include both 
        # 1. camera_frame for video_output
        # 2. websocket template data of the result. 
        super().send_to_component(camera_frame=rgbim,
                                  camera_frame_width=data.width,
                                  camera_frame_height=data.height,
                                  websocket_message=template_data,
                                  websocket_datatype=DATATYPE_TEMPLATE_DATA
                                  )

    def build_template_data(self, text: str, image: bytes, audio_path: str) -> template_data_pb2.TemplateData:
        template_data_proto = template_data_pb2.TemplateData(
            text=text,
            image=image,
            audio_path=audio_path,
        )
        return template_data_proto



    # def _handle_template_request(self, request_data):
    #     _logger.info("Template Request Data: {decoded_data}, {detail}",
    #                  decoded_data=request_data, detail=request_data.detail)
    #     try:
    #         label, image = self._get_detected_label_and_image()
    #         self._send_websocket_template_data(text=label, image=image, audio_path="two_beep_audio")
    #     except Exception:
    #         # No object is detected
    #         _logger.error("Error getting detected label and image")

    # def _send_websocket_template_data(self, text: str = "", image: bytes = None, audio_path: str = "") -> None:
    #     '''
    #     Sending websocket data to the template scene

    #     :param text: text to be displayed
    #     :param image: bytes of the image
    #     :param audio_path: the audio file name without the extension.
    #         The audio file should be in the "Assets/Resources/Audio" folder of the Unity Client
    #     :return: None
    #     '''
    #     websocket_template_data = build_template_data(text=text, image=image, audio_path=audio_path)

    #     super().send_to_component(websocket_message=websocket_template_data)

    #     _logger.info("Sending Template Data (Text: {text}, Image, Audio: {audio_path}) sent to Template Scene",
    #                  text=text, image=image, audio_path=audio_path)

    # # Set the camera frame, frame width, frame height, last detection and class labels in the "shared" memory
    # def _handle_camera_data(self, raw_data: dict) -> None:
    #     super().set_memory_data(base_keys.CAMERA_FRAME, raw_data[base_keys.CAMERA_FRAME])
    #     super().set_memory_data(base_keys.CAMERA_FRAME_WIDTH, raw_data[base_keys.CAMERA_FRAME_WIDTH])
    #     super().set_memory_data(base_keys.CAMERA_FRAME_HEIGHT, raw_data[base_keys.CAMERA_FRAME_HEIGHT])
    #     super().set_memory_data(base_keys.YOLOV8_LAST_DETECTION, raw_data[base_keys.YOLOV8_LAST_DETECTION])
    #     super().set_memory_data(base_keys.YOLOV8_CLASS_LABELS, raw_data[base_keys.YOLOV8_CLASS_LABELS])

    # # Get frame data from memory and get the detected label and image
    # def _get_detected_label_and_image(self) -> tuple:
    #     frame_detections: dict = super().get_memory_data(base_keys.YOLOV8_LAST_DETECTION)
    #     frame: numpy.ndarray = super().get_memory_data(base_keys.CAMERA_FRAME)
    #     frame_width: int = super().get_memory_data(base_keys.CAMERA_FRAME_WIDTH)
    #     frame_height: int = super().get_memory_data(base_keys.CAMERA_FRAME_HEIGHT)
    #     class_labels: dict = super().get_memory_data(base_keys.YOLOV8_CLASS_LABELS)

    #     label, image = self._get_first_yolov8_detection(frame_detections, frame, frame_width, frame_height,
    #                                                     class_labels)
    #     return label, image

    # # Get the first detected label and image from the frame detections
    # def _get_first_yolov8_detection(self, frame_detections: dict, frame: numpy.ndarray, frame_width: int,
    #                                 frame_height: int, class_labels: dict) -> tuple:
    #     detections = YoloDetector.get_detection_in_region(frame_detections, [0, 0, frame_width, frame_height])

    #     if detections is None or len(detections.class_id) == 0:
    #         return None, None

    #     # take the first detection
    #     class_id: int = detections.class_id[0]
    #     xy_bounds: list = detections.xyxy[0]
    #     label: str = class_labels[class_id]

    #     # Crop the frame to the bounding box of the detected object and /
    #     # convert it to bytes before sending it to the template scene
    #     image_frame: numpy.ndarray = image_utility.get_cropped_frame(frame, math.floor(xy_bounds[0]),
    #                                                                  math.floor(xy_bounds[1]),
    #                                                                  math.floor(xy_bounds[2]),
    #                                                                  math.floor(xy_bounds[3]))
    #     image: bytes = image_utility.get_png_image_bytes(image_frame)

    #     return label, image



    # #### Build Data (Protobuf) ######

    # def build_template_data(text: str, image: bytes, audio_path: str) -> template_data_pb2.TemplateData:
    #     template_data_proto = template_data_pb2.TemplateData(
    #         text=text,
    #         image=image,
    #         audio_path=audio_path,
    #     )

    #     return template_data_proto
