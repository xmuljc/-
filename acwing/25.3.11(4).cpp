//��������ֻ�����򣬲���Ҫ�߾��ȣ�ע�⣺�ȱȽ����ַ����Ĵ�С���ٱȽ��ֵ��� 

#include<iostream>
#include<algorithm>
#include<vector>

using namespace std;

struct president{
	int id;
	string str;
	bool operator<(const struct president&other)const{
	 if(str.size()!=other.str.size())return str.size()>other.str.size();
	 else return str>other.str;
	}
	
}p[21];

int main (){
	int n;
	cin>>n;
	string str;
	for(int i=1;i<=n;i++){
		cin>>str;
		p[i].id=i;
		p[i].str=str;
	}
	sort(p+1,p+n+1);
	cout<<p[1].id<<endl<<p[1].str<<endl;
	return 0;
}
