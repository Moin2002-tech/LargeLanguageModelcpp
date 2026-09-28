//
// Created by moinshaikh on 9/28/26.
//
#include<doctest.hpp>

#include<imgui.h>
#include<iostream>
#include<GLFW/glfw3.h>
TEST_CASE("helloGUI") {
    std::cout<<ImGui::GetVersion()<<"\n";
    std::cout<<glfwInit();
}