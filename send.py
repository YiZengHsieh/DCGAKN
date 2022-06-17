import time
ang_data=[[112,39,61,10]]

def send_x(value,path):
	a=open(path,'w')
	a.write(value)
	a.close()

send_x("-F5-145-F2-%s-F3-0"%ang_data[0][1],'E:/arduino/servo/controll_test_xyz.txt')
time.sleep(2)
send_x("F1-%s"%ang_data[0][0],'E:/arduino/servo/controll_test.txt')
time.sleep(2)
send_x("F1-%s-F4-%s"%(ang_data[0][0],ang_data[0][2]),'E:/arduino/servo/controll_test.txt')
time.sleep(2)
send_x("%s"%ang_data[0][3],'E:/arduino/servo/controll_test_arm_move.txt')
'''
for send in range(0,4):
	if send==0:
		#二頭肌旋轉
		if ang_data[0][1]<=80 and ang_data[0][1]>=40:
			a=open('E:/arduino/servo/controll_test_xyz.txt','w')
			a.write("-F5-145-F2-%s-F3-0"%ang_data[0][1])
			a.close()
			print(ang_data[0][1])
	if send==1:
		#二頭肌
		if ang_data[0][0]<=130 and ang_data[0][0]>=85:
			b=open('E:/arduino/servo/controll_txt','w')
			b.write("F1-%s"%ang_data[0][0])
			b.close()
			print(ang_data[0][0])
	if send==2:
		#手肘
		if ang_data[0][2]<=70 and ang_data[0][2]>=55:
			c=open('E:/arduino/servo/controll_txt','w')
			c.write("F1-%s-F4-%s"%(ang_data[0][0],ang_data[0][2]))
			c.close()
			print(ang_data[0][2])
	if send==3:
		#移動平台
		if ang_data[0][3]<=20 and ang_data[0][3]>=7:
			f=open('E:/arduino/servo/controll_test_arm_move.txt','w')
			f.write("%s"%ang_data[0][3])
			f.close()
			print(ang_data[0][3])
	time.sleep(5)
	print(send)
'''