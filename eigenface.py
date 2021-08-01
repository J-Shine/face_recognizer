import matplotlib.pyplot as plt
import numpy as np
import os
import random

"""It helps visualising the portraits from the dataset."""
def plot_portraits(images, titles, h, w, n_row, n_col):
    """
    :param images:
    :param titles:
    :param h:
    :param w:
    :param n_row:
    :param n_col:
    :return:
    """
    """x축 방향으로 2 * n_col, y축 방향으로 2 * n_row 만큼의 크기로 figure 크기 설정(단위 inch)"""
    plt.figure(figsize=(2 * n_col, 2 * n_row))
    """ 그림들의 간격 설정"""
    plt.subplots_adjust(bottom=0, left=0, right=1, top=1)
    """ n_row * n_col 만큼 subplot을 만들어 이미지를 보여준다."""
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
    plt.show()
def plot_portraits_pick(images, titles, h, w, n_row, n_col, points_by_person):

    """x축 방향으로 2 * n_col, y축 방향으로 2 * n_row 만큼의 크기로 figure 크기 설정(단위 inch)"""
    plt.figure(figsize=(2 * n_col, 2 * n_row))
    """ 그림들의 간격 설정"""
    plt.subplots_adjust(bottom=0, left=0, right=1, top=1)
    """ n_row * n_col 만큼 subplot을 만들어 이미지를 보여준다."""
    for i in points_by_person:
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
    plt.show()

"""lfwcrop_grey/faces에서 12288개를 읽음"""
dir='lfwcrop_grey/faces'
celebrity_photos=os.listdir(dir)[0:12288]
celebrity_images=[dir+'/' + photo for photo in celebrity_photos]
images=np.array([plt.imread(image) for image in celebrity_images], dtype=np.float64)
celebrity_names=[name[:name.find('0')-1].replace("_", " ") for name in celebrity_photos]
n_samples, h, w = images.shape
print("Original images")
print("n_samples: {}".format(n_samples))
print("h: {}".format(h))
print("w: {}".format(w))
plot_portraits(images, celebrity_names, h, w, n_row=1, n_col=1)



def pca(X, n_pc):
    """Find mean"""
    mean = np.mean(X, axis=0)
    """Center data"""
    centered_data = X - mean
    """SVD"""
    U, S, V = np.linalg.svd(centered_data)
    """Eigenfaces"""
    components = V[:n_pc]

    return components, mean, centered_data

n_components = 1024
X = images.reshape(n_samples, h * w)

"""
C: components(eigenface)
M: mean
Y: centered_data
"""

C, M, Y = pca(X, n_pc=n_components)
eigenfaces = C.reshape((n_components, h, w))
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
#plot_portraits_pick(eigenfaces, eigenface_titles, h, w, 1, 3)

"""Coefficients"""
weights = np.dot(Y, C.T)

def reconstruction(Y, C, M, h, w, image_index, weights, X, n_components):

    centered_vector=np.dot(weights[image_index, :], C)

    """plot sum of coefficients"""
    """
    stacked_coefficients = []
    sum = float(0)
    for coefficients in weights[image_index]:
        sum = sum + abs(coefficients)
        #sum = sum + coefficients
        #print(coefficients)
        stacked_coefficients.append(sum)

    x = np.arange(n_components)
    y = stacked_coefficients / sum * 100
    #print(x)
    #print(y)

    #plt.scatter(x,y)
    plt.plot(x,y)
    """
    #plt.show()
    #print(weights[image_index, :])
    recovered_image=(M+centered_vector).reshape(h, w)

    return recovered_image

recovered_images=[reconstruction(Y, C, M, h, w, i, weights, X, n_components) for i in range(len(images))]
#recovered_images=[reconstruction(Y, C, M, h, w, i, weights, X) for i in range(4)]
"""
plt.grid()
plt.xlabel("Eigenface")
plt.ylabel("Weights sum")

plt.show()
"""
plot_portraits(recovered_images, celebrity_names, h, w, n_row=1, n_col=1)



"""tests에서 100개를 읽음"""
num_photos_person = 10
dir='tests'
celebrity_photos=os.listdir(dir)[0:num_photos_person * 10]
celebrity_images=[dir+'/' + photo for photo in celebrity_photos]
images=np.array([plt.imread(image) for image in celebrity_images], dtype=np.float64)
celebrity_names=[name[:name.find('0')-1].replace("_", " ") for name in celebrity_photos]
n_samples, h, w = images.shape

cnt = 0
names_lst = []
for names in celebrity_names:
    cnt = cnt + 1
    if(cnt % 10 == 0):
        names_lst.append(names)

print("names_lst is the following")
print(names_lst)

print("Original images")
print("n_samples: {}".format(n_samples))
print("h: {}".format(h))
print("w: {}".format(w))
plot_portraits(images, celebrity_names, h, w, n_row=10, n_col=num_photos_person)

n_components = 1024
X = images.reshape(n_samples, h * w)
print("Xhere")
print(X)

"""
P: projected
C: components(eigenface)
M: mean
Y: centered_data
"""
def findWeights(X, C, M):
    print("mean")
    print(M)
    centered_data = X - M
    print("centered_data")
    print(centered_data)
    print("C.T")
    print(C.T)
    weights = np.dot(centered_data, C.T)
    return weights

weights = findWeights(X, C, M)
print("weights")
print(weights)
print(weights.shape)

"""mean weights"""
weights_mean = []
for k in range(10):
    weights_mean.append(weights[k : k + num_photos_person].mean(axis=0))
    k = k + num_photos_person
print("weights_mean")
for mean in weights_mean:
    print(mean)

"""distance of weights"""
"""
for idx in range(10):
    print("distance is")
    print(np.linalg.norm(weights[idx] - weights_mean[idx]))
"""
#%%

"""
for i in range(10):
    dist = []
    for j in range(num_photos_person):
        dist.append(cos_sim(weights[j + num_photos_person * i], weights_mean[i]))
    dist = np.array(dist)
    print(dist.mean())
"""


def cos_sim(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

step = range(0,num_photos_person * 10,num_photos_person)
unique_weight_point = []
for k in step:
    slice_point = []
    for slice_start in range(1024 - 1):
        l = []
        for i in range(k,k+num_photos_person):
            for j in range(i + 1, k+num_photos_person):
                l.append(cos_sim(weights[i][slice_start:slice_start + 1],
                                 weights[j][slice_start:slice_start+1]))
        l = np.array(l)
        if(l.mean() > 0.9):
            #print("cosine similarity is 1.0 if slice at {}".format(slice_start))
            slice_point.append(slice_start)
    unique_weight_point.append(slice_point)


print("cosine similarity is 1.0 at following points")
i = 1
for points_by_person in unique_weight_point:
    y = []
    for l in range(len(points_by_person)):
        y.append(i)
    plt.scatter(points_by_person, y, s=10)
    print(points_by_person)
    i = i + 1

plt.show()
"""reconstruction"""
recovered_images2=[reconstruction(Y, C, M, h, w, i, weights, X, n_components) for i in range(len(images))]
plot_portraits(recovered_images2, celebrity_names, h, w, n_row=10, n_col=num_photos_person)








"""tests3에서 10개를 읽음"""
num_photos_person = 1
dir='tests3'
celebrity_photos=os.listdir(dir)[0:num_photos_person * 10]
celebrity_images=[dir+'/' + photo for photo in celebrity_photos]
#random.shuffle(celebrity_images)
images=np.array([plt.imread(image) for image in celebrity_images], dtype=np.float64)
celebrity_names=[name[:name.find('0')-1].replace("_", " ") for name in celebrity_photos]
n_samples, h, w = images.shape

print("Original images")
print("n_samples: {}".format(n_samples))
print("h: {}".format(h))
print("w: {}".format(w))
plot_portraits(images, celebrity_names, h, w, n_row=10, n_col=num_photos_person)

n_components = 1024
X = images.reshape(n_samples, h * w)

weights_test = findWeights(X, C, M)

num_photos_person_old = 10
TF = []
k = 0
# 한 사람의 특징점들을 잡음
for slice_start_person in unique_weight_point:
    # 한 test 이미지 잡음
    TF_person = []
    for j in range(k, k + num_photos_person):
        # 한 이미지가 그 사람의 모든 특징점에서 기존의 set과 70% 이상 sim 일치를 이루면
        cnt = 0.0
        for slice_start in slice_start_person:
            l = []
            for i in range(k * 2,k * 2 + num_photos_person_old):
                l.append(cos_sim(weights[i][slice_start:slice_start + 1],weights_test[j][slice_start:slice_start+1]))
            l = np.array(l)
            if(l.mean() > 0.9):
                cnt = cnt + 1.0
        percentage = cnt / len(slice_start_person)
        if(percentage >= 0.7):
            TF_person.append("True")
        else:
            TF_person.append("False")
    TF.append(TF_person)
    k = k + num_photos_person

print("Identification result is ")
for i in TF:
    cnt = 0.0
    print(i)
    """
    for j in i:
        if(j == "True"):
            cnt = cnt + 1.0
    print("accurate rate is {}".format(cnt * 100 / 5.0))
    """

"""
i = 1
for points_by_person in unique_weight_point:
    y = []
    for l in range(len(points_by_person)):
        y.append(i)
    plt.scatter(points_by_person, y, s=3)
    print(points_by_person)
    i = i + 1

plt.show()
"""
"""reconstruction"""
#recovered_images2=[reconstruction(Y, C, M, h, w, i, weights, X, n_components) for i in range(len(images))]
#plot_portraits(recovered_images2, celebrity_names, h, w, n_row=10, n_col=num_photos_person)










"""tests4에서 10개를 읽음"""
num_photos_person = 10
dir='tests4'
celebrity_photos=os.listdir(dir)[0:num_photos_person]
celebrity_images=[dir+'/' + photo for photo in celebrity_photos]
images=np.array([plt.imread(image) for image in celebrity_images], dtype=np.float64)
celebrity_names=[name[:name.find('0')-1].replace("_", " ") for name in celebrity_photos]
n_samples, h, w = images.shape

print("Original images")
print("n_samples: {}".format(n_samples))
print("h: {}".format(h))
print("w: {}".format(w))
plot_portraits(images, celebrity_names, h, w, n_row=1, n_col=num_photos_person)

n_components = 1024
X = images.reshape(n_samples, h * w)
print("Xhere")
print(X)

"""
P: projected
C: components(eigenface)
M: mean
Y: centered_data
"""
def findWeights(X, C, M):
    print("mean")
    print(M)
    centered_data = X - M
    print("centered_data")
    print(centered_data)
    print("C.T")
    print(C.T)
    weights = np.dot(centered_data, C.T)
    return weights

weights = findWeights(X, C, M)
print("weights")
print(weights)
print(weights.shape)

unique_weight_point = []
slice_point = []
k = 0
for slice_start in range(1024 - 1):
    l = []
    for i in range(k,k+num_photos_person):
        for j in range(i + 1, k+num_photos_person):
            l.append(cos_sim(weights[i][slice_start:slice_start + 1],
                             weights[j][slice_start:slice_start+1]))
    l = np.array(l)
    if(l.mean() > 0.9):
        #print("cosine similarity is 1.0 if slice at {}".format(slice_start))
        slice_point.append(slice_start)
unique_weight_point.append(slice_point)


print("cosine similarity is 1.0 at following points")
i = 1
for points_by_person in unique_weight_point:
    y = []
    for l in range(len(points_by_person)):
        y.append(i)
    plt.scatter(points_by_person, y, s=10)
    print(points_by_person)
    i = i + 1

plt.show()
"""reconstruction"""
recovered_images2=[reconstruction(Y, C, M, h, w, i, weights, X, n_components) for i in range(len(images))]
plot_portraits(recovered_images2, celebrity_names, h, w, n_row=1, n_col=num_photos_person)

for points_by_person in unique_weight_point:
    plot_portraits_pick(eigenfaces, eigenface_titles, h, w, n_row=1, n_col=len(points_by_person), points_by_person)

