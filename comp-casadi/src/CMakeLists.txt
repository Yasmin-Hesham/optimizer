cmake_minimum_required(VERSION 2.6)
PROJECT(  compcasadi )

INCLUDE( /opt/robocomp/cmake/robocomp.cmake )

ROBOCOMP_IDSL_TO_ICE( CommonBehavior GenericBase Laser OmniRobot )
