import numpy as np
import cv2
import csv
from numpy import genfromtxt
#my_data = genfromtxt('4/1.csv', delimiter=',')
#3dbox 0.99 (267, 145) (301, 174)
for m in range(1,2):
	img = cv2.imread('J_output/'+str(m)+'.jpg')
	font = cv2.FONT_HERSHEY_SIMPLEX
	my_data = genfromtxt('point/'+str(m)+'x.csv', delimiter=',')
	my_data_depth = genfromtxt('depth_point/'+str(m)+'.csv', delimiter=',')
	#print(my_data.shape)
	#print(my_data[174,301])
	X1=int(my_data[0])
	Y1=int(my_data[1])
	X2=int(my_data[2])
	Y2=int(my_data[3])
	dx=int((X1+X2)/2)
	dy=int((Y1+Y2)/2)
	count=1
	depth_z=my_data_depth[dy,dx]
	if depth_z!=0:
		print("ID:",m,"___",X1,X2,dx,"___",Y1,Y2,dy,"___",depth_z)
	elif depth_z==0:
		while depth_z==0:
			depth_z=my_data_depth[dy-count,dx-count]
			count+=1
		print("ID:",m,"___",X1,X2,dx,"___",Y1,Y2,dy,"___",depth_z)
	with open('data_output/data3.csv','a',newline='') as f:
		th=csv.writer(f)
		th.writerow([m,dy,dx,int(depth_z/10)])
	'''	
	matrixA=np.zeros((X2-X1+1)*(Y2-Y1+1))
	print(matrixA.shape)
	count_i=0

	for x in range(X1,X2+1):
		for y in range(Y1,Y2+1):
			matrixA[count_i]=my_data_depth[y,x]
			count_i+=1
	
	cv2.putText(img,str((np.mean(matrixA)/10)),(0,30),font,0.5,(0,255,76),2)
	cv2.imwrite('J_output/'+str(m)+'.jpg',img)
	print('depth point:',np.mean(matrixA)/10,'cm') #
	'''

'''
#自動存CSV
print(matrixA[0],matrixA[1049])
print(np.mean(matrixA)) #
matrixB=np.zeros(4)
matrixB[0]=1
matrixB[1]=122
a=matrixB.reshape(1,4)

#f = tables.open_file('x.csv', mode='a')
#f.root.data.append(a)
np.savetxt('x.csv',a, fmt='%s',delimiter = ',')
'''