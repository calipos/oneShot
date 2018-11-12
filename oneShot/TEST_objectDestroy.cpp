#include<iostream>
class B
{
public:
	B() { std::cout << "construct B" << std::endl; };
	~B() { std::cout << "destroy B" << std::endl; };
};

class A
{
public:
	A(int a) { std::cout <<"construct A" << std::endl; };
	B b;
	~A() { std::cout << "destroy A" << std::endl; };

};


void TEST_ogjectDestroy()
{
	A aa(1);
	aa.~A();
}