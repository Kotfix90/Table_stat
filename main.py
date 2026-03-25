import cv2
import pandas as pd
from ultralytics import YOLO
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Прототип детекции уборки столов")
    parser.add_argument("--video", required=True, help="Путь к видеофайлу")
    parser.add_argument("--start_time", type=float, default=0.0, help="Время начала обработки (сек)")
    parser.add_argument("--end_time", type=float, default=None, help="Время окончания обработки (сек)")
    parser.add_argument("--max_duration", type=float, default=None, help="Максимальная длительность (сек) от start_time")
    return parser.parse_args()

def select_table_roi(frame, scale_factor=0.5):
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    print("\n[ИНФО] Выделите столик мышкой и нажмите ENTER. ESC для отмены.")
    roi = cv2.selectROI("Select Table ROI", small_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    if roi == (0, 0, 0, 0):
        print("[ОШИБКА] ROI не выбран. Выход.")
        sys.exit()
    return (int(roi[0]/scale_factor), int(roi[1]/scale_factor), 
            int(roi[2]/scale_factor), int(roi[3]/scale_factor))

def detect_events(video_path, start_time=0.0, end_time=None, max_duration=None):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ОШИБКА] Не удалось открыть {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    
    ret, frame = cap.read()
    if not ret:
        print("[ОШИБКА] Не удалось прочитать видео")
        return
    
    tx, ty, tw, th = select_table_roi(frame)

    # Состояния
    STATE_EMPTY = "Empty"
    STATE_APPROACH = "Approach"
    STATE_OCCUPIED = "Occupied"

    current_state = STATE_EMPTY
    
    # ПЕРЕМЕННЫЕ ДЛЯ АНАЛИТИКИ
    last_exit_timestamp = None  # Момент, когда гость ушел (стол стал пустым)
    cleaning_delays = []        # Список интервалов "уход -> новый подход"
    
    # Настройки
    APPROACH_CONFIRM_SEC = 2.0
    EMPTY_CONFIRM_FRAMES = 25
    
    empty_counter = 0
    events_list = []

    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        curr_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if end_time and curr_sec > end_time: break
        if max_duration and (curr_sec - start_time) > max_duration: break

        results = model(frame, classes=[0], verbose=False)[0]
        person_in_zone = False
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            foot_x, foot_y = (x1 + x2) // 2, y2
            if (tx <= foot_x <= tx + tw) and (ty <= foot_y <= ty + th):
                person_in_zone = True
                break

        # ЛОГИКА ПЕРЕХОДОВ
        if person_in_zone:
            empty_counter = 0
            if current_state == STATE_EMPTY:
                current_state = STATE_APPROACH
                entry_time = curr_sec
                events_list.append({"timestamp": curr_sec, "event": "Approach"})
                
                # Расчёт статистики 
                if last_exit_timestamp is not None:
                    delay = curr_sec - last_exit_timestamp
                    cleaning_delays.append(delay)
                    print(f"[АНАЛИТИКА] Время между уходом и подходом: {delay:.2f} сек.")
            
            elif current_state == STATE_APPROACH:
                if (curr_sec - entry_time) >= APPROACH_CONFIRM_SEC:
                    current_state = STATE_OCCUPIED
                    events_list.append({"timestamp": curr_sec, "event": "Occoupied"})
        else:
            empty_counter += 1
            if empty_counter >= EMPTY_CONFIRM_FRAMES:
                if current_state != STATE_EMPTY:
                    # Момент фактического перехода в пустой статус (уход гостя)
                    current_state = STATE_EMPTY
                    last_exit_timestamp = curr_sec 
                    events_list.append({"timestamp": curr_sec, "event": "Empty"})

        # Визуализация
        color = (0, 255, 0) if current_state == STATE_EMPTY else (0, 255, 255) if current_state == STATE_APPROACH else (0, 0, 255)
        cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), color, 3)
        cv2.putText(frame, f"STATUS: {current_state}", (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        out.write(frame)
        cv2.imshow("Monitor", cv2.resize(frame, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Итоговый расчет
    avg_cleaning_time = sum(cleaning_delays) / len(cleaning_delays) if cleaning_delays else 0
    
    print(f"\n--- СТАТИСТИКА УБОРКИ ---")
    print(f"Количество циклов смены гостей: {len(cleaning_delays)}")
    print(f"Среднее время между уходом и новым подходом: {avg_cleaning_time:.2f} сек.")
    
    # Сохранение лога и отчета
    df = pd.DataFrame(events_list)
    df.to_csv("events_log.csv", index=False)
    
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(f"Среднее время между уходом гостя и подходом следующего: {avg_cleaning_time:.2f} сек.\n")
        f.write(f"Всего циклов: {len(cleaning_delays)}\n\n")
        f.write(df.to_string())

if __name__ == "__main__":
    args = parse_args()
    detect_events(args.video, args.start_time, args.end_time, args.max_duration)
    