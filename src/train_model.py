import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle # thư viện lưu dữ liệu của models vào


#TODO : step 1 xem qua datasets và xử lý các phần bị khuyết
# =============================================================

data= pd.read_csv("../data/co2.csv")
data["time"]=pd.to_datetime(data["time"],dayfirst=False)
print("------")
print(data.info())
data["co2"]= data["co2"].interpolate() # nội suy ra các điểm khuyết

#TODO :step2 tạo 1 hàm để vẽ thêm matrix từ datasets ban đầu
# =============================================================

def convert_data(raw_data,predictors_lags=5):
    i=1
    while i < predictors_lags:
        raw_data["co2_{}".format(i)]=raw_data["co2"].shift(-i) # tạo thêm cột mới dựa vào cột datasets ban đầu
        i+=1
    raw_data["target"]=raw_data["co2"].shift(-i) # dịch thêm lần nữa để có cột target
    raw_data.dropna(axis=0)
    return raw_data

#TODO:step3 visualize biểu đồ datasets để xem trước
# =============================================================

# fig,(ax,ax1)=plt.subplots(2) # tạo bàn và đĩa
#
# plt.subplots_adjust(hspace=0.5) # nghịch
# fig.suptitle('Time Series Forecasting') # nghịch
#
# ax.plot(datasets["time"],datasets["co2"]) # lấy dữ liệu x và y để vẽ
# ax.set_title("Co2 Forecast") # nghịch
# ax1.set_title("Test") # nghịch
# ax.set_xlabel("Time") # gán tên trục Time đại diện cho trục x
# ax.set_ylabel("CO2") #  gán tên trục CO2 đại diện cho trục y
# plt.show()
#TODO:step4 chia dữ liệu matrix theo chiều ngang và dọc
# =============================================================

new_data= convert_data(data,5)
new_data= new_data.dropna(axis=0)
x= new_data.drop(["time","target"],axis=1)
y= new_data["target"]

train_ratio=0.8
num_samples=len(x)
x_train= x[:int(train_ratio*num_samples)]
y_train= y[:int(train_ratio*num_samples)]
x_test= x[int(train_ratio*num_samples):]
y_test= y[int(train_ratio*num_samples):]

#TODO:step5 tiền xử lý nhưng thôi do dữ liệu đều từ 1 cột mà ra nên nó cùng range rồi
# =============================================================

#     #FIXME: sau đó phải chuẩn hóa cào bằng dữ liệu để models đối xử công bằng với tất caả cột dữ liệu( trừ mô hình RandomForestClassifier hay decision tree thì ko cần chuẩn hóa vì nó ko thao tác trực tiếp với datasets mà chỉ so sánh datasets theo giá trị thật của từng cột thôi)
# scaler= StandardScaler() #tạo ra một đối tượng (scaler) StandardScaler để sau này dùng fit() và transform().
# scaler.fit(x_train) # đo cách chỉ số của từng cột, tính std và kì vọng của từng cột
# x_train= scaler.transform(x_train) # biến đổi giá trị theo std và kì vọng vừa đo đc
# x_test= scaler.transform(x_test) # chỉ cần transform thôi vì bộ test cần biến đổi theo mean và std của bộ train đã được lưu trong scaler
#
#TODO step6 fit mô hình thôi
# =============================================================

model= LinearRegression()
model.fit(x_train,y_train)
y_pred= model.predict(x_test)

for i,j in zip(y_test,y_pred):
    print("y_test:{}".format(i),"y_pred:{}".format(j))

#TODO Step7: in ra thông số() của mô hình
# =============================================================

    #FIXME cách1: in từng giá trị 1
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("neg_mean_absolute_error:", mean_absolute_error(y_test, y_pred))
print("Mean Sqare Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

#TODO: step8 lại mô phỏng tiếp
# =============================================================

fig,ax=plt.subplots() # tạo bàn và đĩa

plt.subplots_adjust(hspace=0.5) # nghịch
fig.suptitle('Time Series Forecasting') # nghịch

ax.plot(new_data["time"][:int(train_ratio*num_samples)],new_data["co2"][:int(train_ratio*num_samples)],label="train") # lấy dữ liệu x và y để vẽ
ax.plot(new_data["time"][int(train_ratio*num_samples):],new_data["co2"][int(train_ratio*num_samples):],label="test") # lấy dữ liệu x và y để vẽ
ax.plot(new_data["time"][int(train_ratio*num_samples):],y_pred,label="prediction") # lấy dữ liệu x và y để vẽ
ax.grid() # tạo đường kẻ 
ax.legend() # tạo ra phần chú thích các đường


# ax.set_title("Co2 Forecast") # nghịch
# ax1.set_title("Test") # nghịch
ax.set_xlabel("Time") # gán tên trục Time đại diện cho trục x
ax.set_ylabel("CO2") #  gán tên trục CO2 đại diện cho trục y
plt.show()
#TODO step9: save the models to disk sau khi fit là dùng đc
# ======================================================
filename = '../models/co2_model.pkl'
pickle.dump(model, open(filename, 'wb'))
