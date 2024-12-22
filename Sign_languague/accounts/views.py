from django.shortcuts import render,redirect

from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.hashers import make_password
from django.http import StreamingHttpResponse
import json
from .models import History
from .models import Courses
from django.http import JsonResponse
from django.utils import timezone
import cv2
import mediapipe as mp
import time
from tensorflow.keras.models import load_model
import numpy as np
import time

# Initialize Model Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
   
# Create your views here.

def RegisterAccount(request):
   if request.method == 'POST':
      first_name = request.POST.get('first_name')
      last_name = request.POST.get('last_name')
      username = request.POST.get('username')
      password = request.POST.get('password')
      confirm_password = request.POST.get('confirm_password')
     
      if password!= confirm_password:
         messages.error(request,'Mật khẩu không khớp!')
         return redirect('signup')
      if User.objects.filter(username=username).exists():
         messages.error(request,'Tên tài khoản đã tồn tại!')
         return redirect('signup')
      # Tạo người dùng mới
      new_user = User(first_name=first_name,last_name=last_name,username=username,password=make_password(password))
      # Lưu người dùng
      new_user.save()
      messages.success(request,'Đăng ký tài khoản thành công')
      login(request,new_user)
      return redirect('home') 
   return render(request,'accounts/sign-up.html')    


def LoginAccount(request):
   
   if request.method == 'POST':
      username = request.POST.get('username')
      password = request.POST.get('password')   
      # Validate infor account
      user = authenticate(request, username=username,password = password)
      if user:
         login(request, user)
         messages.success(request,'Đăng nhập thành công')
         return redirect('home')

      else:
         # Thông báo lỗi nếu thông tin không đúng
         messages.error(request,'Tên tài khoản hoặc mật khẩu không chính xác!')
         return render(request, 'accounts/login.html')
   
   return render(request,'accounts/login.html')

@login_required
def HomePage(request):
   
   return render(request,'accounts/homepage.html')

@login_required
def logout_view(request):
   logout(request)
   return redirect('login')

import cv2
from django.http import StreamingHttpResponse

def generate_frames(request):
   # Mở camera với ID được chọn
   cap = cv2.VideoCapture(1)
   dict_label = {
    0: 'Hello',
    1: 'I', #Goodbye
    2: 'Yes',
    3: 'No',
    # 4: 'Bad',
    5: 'My',
    6: 'What', #Thanks,
    7: 'Sorry',
    8: 'I/Me',
    9: 'You',
    10: 'A',
    11: 'B',
    12: 'C',
    13: 'D',
    14: 'E',
    15: 'F',
    16: 'G',
    17: 'H',
    18: 'I',
    19: 'J',
    20: 'K',
    21: 'L',   
    22: 'M',
    23: 'N',
    24: 'O',
    25: 'P',
    26: 'Q',
    27: 'R',
    28: 'S',
    29: 'T',
    30: 'U',
    31: 'V',
    32: 'W',
    33: 'X',
    34: 'Y',
    35: 'Z',
    36:'What',
    37:'Eat',
    38:'My',
    39:'Name'
}


   mp_hands = mp.solutions.hands
   mp_drawing = mp.solutions.drawing_utils

   hands = mp_hands.Hands(
      static_image_mode = False,
      max_num_hands = 1,
      min_detection_confidence = 0.4,
      min_tracking_confidence = 0.1
      
   )

   model = load_model('model-LTSM-ver4.h5')
   frame_each_video = 30
   sequences_frame =[]
   check = 0
   while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
         break
      
      each_frame = []
      # Tạo bản sao frame để xử lý
      # processed_frame = frame.copy()
      frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      results = hands.process(frame_rgb)
      if results.multi_hand_landmarks:
         for hand_landmarks in results.multi_hand_landmarks:
         # Define landmark and draw in hands  
         
            mp_drawing.draw_landmarks(
               frame, 
               hand_landmarks,
               mp_hands.HAND_CONNECTIONS,
               mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
               mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
         
            x_=[]
            y_=[]
            z_ = []
            for i in range(len(hand_landmarks.landmark)):
               x = hand_landmarks.landmark[i].x
               y = hand_landmarks.landmark[i].y
               z = hand_landmarks.landmark[i].z
               x_.append(x)
               y_.append(y)
               z_.append(z)
            h,w,_ = frame.shape
            # print(h,w,_)
            pos_text=(int(min(x_)*w)-120,int(min(y_)*h)+60)
            cv2.rectangle(frame,(int(min(x_)*w)-20,int(min(y_)*h)-20),(int(max(x_)*w)+20,int(max(y_)*h)+20),(0,255,0),2)
         
            x_ = [ele - min(x_) for ele in x_]
            y_ = [ele - min(y_) for ele in y_]
            # z_ = [ele - min(z_) for ele in z_]
         each_frame.extend(x_)
         each_frame.extend(y_)
         each_frame.extend(z_)
         if len(each_frame) == 21*3:
         # each_frame.extend([0]*(21*2-len(each_frame)))
            sequences_frame.append(each_frame)
         
         if len(sequences_frame) == frame_each_video:
            sequences_frame1 = np.asarray([sequences_frame])
            y_pred = model.predict(sequences_frame1)
            # print('y_pred1',y_pred)
            max_rate = np.max(y_pred)
            print('MAX RATE: ',max_rate)
            y_pred = np.argmax(model.predict(sequences_frame1),axis=1)
            print(dict_label[y_pred[0]])
            
            if max_rate >=0.75:
               cv2.putText(frame,f'{dict_label[y_pred[0]]}:{max_rate:.2f}' ,pos_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3,
                     cv2.LINE_AA)
               print(dict_label[y_pred[0]])
               # Save predict in model History
               user = request.user
               History.objects.create(user = user,word = dict_label[y_pred[0]])
               check = 1
            sequences_frame=[]

         

      # Chuyển frame gốc sang định dạng JPEG để hiển thị mà không thay đổi nội dung
      ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
      frame = buffer.tobytes()

      # Truyền frame gốc (không qua xử lý) đến trình duyệt
      yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
      
      if check == 1:
         yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
         cv2.waitKey(2000)
         check = 0
         # time_start = time.time()
         # while True:
         #    if time.time() - time_start < 0.1:
         
         # response =  StreamingHttpResponse(b'--frame\r\n'
         #       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
         # response['X-condition-Met']='true'
         # yield response
         
         
         # check = 0
         
         
      # if check == 1:
      #    cv2.waitKey(3000)
      #    check = 0
   cap.release()

def generate_frame2(request):
   actions = ['hello','goodbye','yes','thankyou','my','name','I_me','you','nice','meet']
   sequence = []
   sentence = []
   threshold = 0.9
   
   model = load_model("action3.h5")
   
   # Predict in Real-time
   cap = cv2.VideoCapture(1)
   if not cap.isOpened():
      exit()
   with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
      count = -60
      while cap.isOpened():
         # Read frame
         ret, frame = cap.read()
         if not ret:
            break
         count +=1
         # Get landmarks in frame
         image, results = mediapipe_detection(frame, holistic)
         # print(results)
         
         # Draw landmarks
         draw_style_landmarks(image, results)
         # Extract keypoints each frame
         keypoints = extract_keypoints(results=results)
         # Append frame to sequence
         sequence.append(keypoints)
         sequence = sequence[-60:]
         # print(len(sequence))
         
         if count == 60:
            res = model.predict(np.expand_dims(sequence,axis=0))[0]
            print(res)
            print(actions[np.argmax(res)])
            
            if res[np.argmax(res)] > threshold:
               if len(sentence) > 0:
                  if actions[np.argmax(res)] != sentence[-1]:
                     sentence.append(actions[np.argmax(res)])
                     # Add new translate-history
                     user = request.user
                     History.objects.create(user = user,word =actions[np.argmax(res)])
               
               else:
                  sentence.append(actions[np.argmax(res)])
            if len(sentence) > 4:
               sentence = sentence[-4:]
         
            count = -60
         cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
         cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
         cv2.putText(image, f'Frame: {count}', (3,70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)   
         # Display frame
         # cv2.imshow('Frame',image)
         
         # Short break
         if cv2.waitKey(10) & 0xFF == ord('q'):
            break
         
         # Chuyển frame gốc sang định dạng JPEG để hiển thị mà không thay đổi nội dung
         ret, buffer = cv2.imencode('.jpg', cv2.flip(image,1))
         frame = buffer.tobytes()

         # Truyền frame gốc (không qua xử lý) đến trình duyệt
         yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
      cap.release()
      cv2.destroyAllWindows()
   
# Function to detect landmarks in frame
def mediapipe_detection(image, model):
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   image.flags.writeable = False
   results = model.process(image)
   image.flags.writeable = True
   image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
   return image, results

# Function to draw Landmarks and connection of landmarks
def draw_style_landmarks(image, results):
   # Draw face landmarks
   # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
   #                         mp_drawing.DrawingSpec(color = (80,110,10), thickness = 1, circle_radius = 1),
   #                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
   #                         )
   # Draw pose connections
   mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                           mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                           mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                           ) 
   # Draw left hand connections
   mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                           mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                           mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                           ) 
   # Draw right hand connections  
   mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                           mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                           mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                           ) 
   
# Function to extract landmarks
def extract_keypoints(results):
   # Pose has 33 landmarks
   # Shape (33,4)->(132,)
   pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
   # Face has 468 landmarks
   # Shape (468,3)-> (1404, )
   # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
   # Left-hand has 21 landmarks
   # Shape (21,3)-> (63, )
   lh = np.array([[res.x, res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
   # Right-hand has 21 landmarks
   # Shape (21,3)-> (63, )
   rh = np.array([[res.x, res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

   # return np.concatenate([pose, face, lh, rh]) # Shape (1662,) [......]
   return np.concatenate([pose, lh, rh]) # Shape (258,) [......]


def video_feed(request):
   # View để stream video
   return StreamingHttpResponse(generate_frame2(request), content_type='multipart/x-mixed-replace; boundary=frame')
 
def get_user_history(request):
   user = request.user
   histories = History.objects.filter(user=user).order_by('-timestamp')[:10]  # Lấy 10 bản ghi mới nhất
   history_list = [
      {"word": history.word, 
       "timestamp": timezone.localtime(history.timestamp,timezone.get_fixed_timezone(420)).strftime("%d-%m-%Y %H:%M")
      } 
      for history in histories
   ]
   return JsonResponse(history_list, safe=False)

# Get record in Courses table
def get_courses(request):
   courses = Courses.objects.all()
   course_list = [
      {
         "image": course.image.url,
         "title":course.title,
         'rating': course.rating,
         'reviews':course.reviews,
         'start_date': course.start_date.strftime('Ngày %d tháng %m năm %Y'),
         'time':course.time,
         'sale_price':course.sale_price,
         'original_price': course.original_price
      }
      for course in courses
   ]
   return JsonResponse(course_list,safe =False)

# Courses
@login_required
def courses_view(request):
   if request.method == 'POST':
      messages.success(request,'Gửi thông tin thành công')
      
   return render(request,'accounts/courses.html')