// RayAABB.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include "float.h""

#include "box.h"
#include "ray.h"
#include "vector3.h"


int main()
{
	Box block(Vector3(0.5, 0.5, 0.5) , Vector3(1, 1, 1));
	Ray ray(Vector3(0,0,0), Vector3(1,1,1));
	if (block.intersect(ray, 0.0, 0.5)){
		std::cout << "Intersect" << std::endl;
	}else{
		std::cout << "No intersection" << std::endl;
	}

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
