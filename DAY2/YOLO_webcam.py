import time
import math
import os
import sys
# [삭제] import rclpy, Node, Image, CvBridge (원본: ROS2 관련 라이브러리들)
import threading
from queue import Queue
from ultralytics import YOLO
from pathlib import Path
import cv2

# [수정] 일반 클래스로 변경 | (원본: class YOLOImageSubscriber(Node):)
class YOLOWebcamSubscriber: 
    def __init__(self, model):
        # [삭제] super().__init__('yolo_image_subscriber') (원본: ROS 노드 이름 설정)
        self.model = model
        # [삭제] self.bridge = CvBridge() (원본: ROS 이미지를 변환하던 도구)
        self.image_queue = Queue(maxsize=1)
        self.should_shutdown = False
        self.classNames = model.names if hasattr(model, 'names') else ['Object']

        # [추가] 웹캠 연결 설정
        self.cap = cv2.VideoCapture(0) 

        # [수정] 웹캠 캡처 스레드 추가 | (원본: self.subscription = self.create_subscription(...))
        # ROS는 시스템이 사진을 던져줬지만, 웹캠은 내가 직접 찍는 일꾼이 필요합니다.
        self.capture_thread = threading.Thread(target=self.webcam_capture_loop, daemon=True)
        self.capture_thread.start()

        # AI 분석 스레드는 그대로 유지
        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()

    # [수정] 함수명 및 방식 변경 | (원본: def listener_callback(self, msg):)
    def webcam_capture_loop(self):
        while not self.should_shutdown:
            # [수정] 웹캠에서 직접 읽기 | (원본: img = self.bridge.imgmsg_to_cv2(msg, ...))
            ret, img = self.cap.read()
            if not ret:
                continue
            
            if not self.image_queue.full():
                self.image_queue.put(img)
            else:
                # 큐가 꽉 찼을 때 최신화를 위해 하나 비우고 넣기 (지연 방지)
                try:
                    self.image_queue.get_nowait()
                    self.image_queue.put(img)
                except:
                    pass

    def detection_loop(self):
        # 이 루프 내부 로직은 원본과 거의 동일합니다.
        while not self.should_shutdown:
            try:
                img = self.image_queue.get(timeout=0.5)
            except:
                continue

            results = self.model.predict(img, stream=True)

            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0]) if box.cls is not None else 0
                    conf = float(box.conf[0]) if box.conf is not None else 0.0

                    label = f"{self.classNames[cls]} {conf:.2f}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # [수정] 로거 대신 print 사용 | (원본: self.get_logger().info(...))
                print("Shutdown requested via 'q'") 
                self.should_shutdown = True
                break

def main():
    model_path = input("Enter path to model file (.pt): ").strip()

    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        sys.exit(1)

    model = YOLO(model_path)

    # [삭제] rclpy.init() (원본: ROS 시스템 초기화)
    
    # [수정] 클래스 생성 | (원본: node = YOLOImageSubscriber(model))
    processor = YOLOWebcamSubscriber(model) 

    try:
        # [수정] 일반 while 루프로 변경 | (원본: while rclpy.ok() and not node.should_shutdown: rclpy.spin_once(...))
        while not processor.should_shutdown:
            time.sleep(0.1) 
    except KeyboardInterrupt:
        print("Shutdown requested via Ctrl+C.")
    finally:
        processor.should_shutdown = True
        processor.cap.release() # [추가] 웹캠 자원 해제
        # [삭제] node.destroy_node(), rclpy.shutdown() (원본: ROS 노드 종료)
        cv2.destroyAllWindows()
        print("Shutdown complete.")

if __name__ == '__main__':
    main()