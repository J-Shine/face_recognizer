# face_recognizer

## Full Pipeline

<img width="670" alt="pipeline" src="https://user-images.githubusercontent.com/61873510/127987299-72c242cf-e5bb-4857-b29e-33e9c9105bed.png">

```Training Set```: 12288 64x64 grey face images of people.<br>
```Validation Set(Training Set 2)```: 100 64x64 grey face images of 10 people, a cat and a dog.<br>
```Test Set```: 50 64x64x grey face images of 10 people, a cat and a dog.<br>
```PCA```: Principal Component Analysis<br>
```Unique face points```: points with cosine similarity 1.0.<br><br>
## Steps
<img width="670" alt="Step one" src="https://user-images.githubusercontent.com/61873510/127992770-385dff8b-1c9f-4655-9a31-4a6e684e7d96.png">
First, extract eigenfaces using PCA and pick 1024 most weighted eigenfaces which reconstruct images perfectly when seen with the eyes.<br><br><br>

<img width="670" alt="Step Two" src="https://user-images.githubusercontent.com/61873510/127992788-7ed691b1-7107-4b2f-9c16-1cf0ecbdc4d6.png">
Second, get cosine similarities on the extracted eigen faces on the first step and the vaildation set. Choose points which has 1.0 score on cosine similarity.<br><br><br>

<img width="670" alt="Step Three" src="https://user-images.githubusercontent.com/61873510/127992800-cb31112c-229a-46d0-a125-9c7afb4ac5ff.png">
Third, try to recognize each face images of the test set by getting cosine simlarity of unique points and the test set.<br><br><br>

## Result

<img width="670" alt="Test Result" src="https://user-images.githubusercontent.com/61873510/127989800-077b1a81-29cf-486d-9072-f2ca985d8bed.png">

```Result```: 38% recognition rate.

