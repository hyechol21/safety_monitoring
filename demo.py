import threading
from collections import deque
import cv2
import time

import torch
import track
from lib import yolo
from classify import Classify


class ThreadInput(threading.Thread):
    def __init__(self, url=None):
        threading.Thread.__init__(self)
        self.cap = None
        self.frame = None
        self.frame_cnt = 0
        self.results = dict()
        self.fps = 0

        if url is not None:
            self.width, self.height = 1080, 720
            self.cap = cv2.VideoCapture(url)
            self.cap.set(3, self.width)
            self.cap.set(4, self.height)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def run(self):
        while flag:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, dsize=(self.width, self.height), interpolation=cv2.INTER_AREA)
            self.frame = frame
            self.frame_cnt += 1

            if (self.frame_cnt % 3 == 0) and len(th_detect.deque) < 50:
                th_detect.deque.append(frame)
            time.sleep(1/self.fps)
        self.cap.release()


class ThreadClassify(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.deque = deque()
        self.classify = Classify(model_name=model_name, weight=weight, labels_map=labels_map)
        self.tracker = dict() # id당 최대 5개
        self.unmmatched_track = dict() # id가 5번 유실된 경우 tacker에서 제거

    def run(self):
        while flag:
            if len(self.deque) > 0:
                detections = self.deque.popleft()
                outputs = detections[0]
                frame = detections[1]

                temp_track = [x for x in self.tracker.keys()]

                for track in outputs:
                    xmin, ymin, xmax, ymax, uid = track
                    # 박스 길이 늘리기
                    if ymin-10 < 0:
                        ymin = 0
                    else:
                        ymin = ymin-10

                    if xmin-10 < 0:
                        xmin = 0
                    else:
                        xmin = xmin-10

                    roi = frame[ymin:ymax, xmin:xmax]

                    result_classify, result_prob = self.classify.predict_image_cv2(roi)
                    print(f'{uid}: {result_classify}    ({result_prob})')

                    # if result_prob < 0.7:
                    #     continue

                    # 소실되었던 uid가 다시 검출된 경우
                    if uid in self.unmmatched_track:
                        del self.unmmatched_track[uid]

                    if uid not in self.tracker:
                        self.tracker[uid] = deque()
                        self.tracker[uid].append(result_classify[0])
                    elif len(self.tracker[uid]) < 5:
                        temp_track.remove(uid)
                        self.tracker[uid].append(result_classify[0])
                    else:
                        temp_track.remove(uid)
                        self.tracker[uid].popleft()
                        self.tracker[uid].append(result_classify[0])

                # tracker에는 있으나 세로 detect된 uid에는 없는 경우
                for uid in temp_track:
                    if uid not in self.unmmatched_track:
                        self.unmmatched_track[uid] = 1
                    else:
                        self.unmmatched_track[uid] += 1

                    if self.unmmatched_track[uid] >= 5:
                        del self.tracker[uid]
                        del self.unmmatched_track[uid]
            time.sleep(0.0001)


class ThreadDetect(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.deque = deque()

    def run(self):
        while flag:
            if len(self.deque) > 0:
                frame = self.deque.popleft()
                detections = yolo.net.detect(frame)
                if not detections:
                    th_read.results = None
                    continue
                results = dict()
                label = []
                bbox_xywh = []
                confs = []
                for i, detection in enumerate(detections):
                    if detection[0] == 'person' or detection[0] == 'falldown':
                        print(detection)
                        confs.append(detection[1])
                        bbox_xywh.append(detection[2])
                try:
                    outputs = track.deepsort.update((torch.Tensor(bbox_xywh)), (torch.Tensor(confs)), frame)
                except:
                    print(bbox_xywh)
                results["label"] = label
                results["outputs"] = outputs

                th_read.results = results
                if len(th_classify.deque) < 50:
                    th_classify.deque.append([outputs, frame])
            time.sleep(0.0001)


class ThreadOutput(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global flag
        frame_cnt = 0
        start_time = time.time()
        while True:
            if th_read.frame is not None:
                frame = th_read.frame.copy()

                if th_read.results and len(th_read.results.keys()):
                    detections = th_read.results["outputs"]

                    for i, detection in enumerate(detections):
                        xmin, ymin, xmax, ymax, uid = detection
                        label = ''

                        if ymin - 10 < 0:
                            ymin = 0
                        else:
                            ymin = ymin - 10

                        if xmin - 10 < 0:
                            xmin = 0
                        else:
                            xmin = xmin - 10

                        if uid in th_classify.tracker:
                            if len(th_classify.tracker[uid]) >= 5:
                                label_idx = max(th_classify.tracker[uid])
                                label = labels_map[label_idx]

                        if uid is not None:
                            color = track.compute_color_for_labels(uid)
                            pstring = '{}{:d} : {}'.format("", uid, label)
                        else:
                            color = (255,255,255)
                            pstring = ''
                        # pstring = label + ": " + str(np.rint(100 * confidence)) + "%"
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        cv2.putText(frame, pstring, (xmin, ymin - 12), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)
                cv2.imshow('output', frame)
                frame_cnt += 1

            if time.time() - start_time >= 1:
                fps = frame_cnt
                # print('FPS: ', fps)
                frame_cnt = 0
                start_time = time.time()

            if cv2.waitKey(1) & 0xff == ord('q'):
                flag = False
                break
            time.sleep(0.01)
        cv2.destroyAllWindows()


if __name__=='__main__':
    cctv = './video/1.avi'
    # define classify
    labels_map = ["helmet", "no helmet", "unknown"]

    model_name = 'efficientnet-lite2'
    weight = './weights/helmet.pt'

    flag = True

    th_classify = ThreadClassify()
    th_classify.start()

    th_detect = ThreadDetect()
    th_detect.start()

    th_read = ThreadInput(cctv)
    th_read.start()

    th_view = ThreadOutput()
    th_view.start()
