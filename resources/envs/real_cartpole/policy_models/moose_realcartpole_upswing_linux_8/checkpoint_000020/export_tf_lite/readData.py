#!/usr/bin/python
# -*- coding: UTF-8 -*-

# from cartpole_fuzzyGP_raspberry import run as gpFuzzyRun
# from cartGP import CartpoleFuzzyGP
from CartpoleGeneticProgrammingController import CartpoleGeneticProgrammingController
from CartpoleTFLitePolicyController import CartpoleTFLitePolicyController
from CartpoleTFPolicyController import CartpoleTFPolicyController
from CartpoleTFVQCPolicyController import CartpoleTFVQCPolicyController
from CartpoleTFVQCPolicyControllerTorch import CartpoleTFVQCPolicyControllerTorch
# from CartpoleTFCircuitVQCPolicyController import CartpoleTFCircuitVQCPolicyController
from CartpoleTFLiteBSFController import CartpoleTFLiteBSFController
from CartpolePSOPolicyController import CartpolePSOPolicyController

from TestCartpoleTFCircuitVQCPolicyController import CartpoleTFCircuitVQCPolicyController
from TestCartpoleTFLitePolicyController import TestCartpoleTFLitePolicyController
from TestCartpoleTFCSupervisedPolicyController import TestCartpoleTFCSupervisedPolicyController
from TestCartpoleCLSupervisedPolicyController import TestCartpoleCLSupervisedPolicyController
from TestCartpoleTFCPolicyController import TestCartpoleTFCPolicyController
from TestCartpoleBNNTFLitePolicyController import TestCartpoleBNNTFLitePolicyController

from gamepad import TGamepad, buttonStatus
from enum import Enum
import serial, time, struct
import socket
from datetime import datetime
import threading
from time import sleep
from threading import Timer
from tm1637 import show_CP_integer, TM1637
from State import State
from random import random
import csv
import operator

cartPosition = 0
pendulumPosition = 0

# configure serial
serCart = serial.Serial()
# ser.port = "/dev/rfcomm1" # Set this to your serial port
# serCart.port = "/dev/ttyUSB2" # Set this to your serial port
serCart.port = "/dev/ttyUSB.cartPosition"  # Set this to your serial port
serCart.baudrate = 9600
serPendulum = serial.Serial()
# serPendulum.port = "/dev/ttyS0" # Set this to your serial port
# serPendulum.port = "/dev/ttyUSB0" # Set this to your serial port
serPendulum.port = "/dev/ttyUSB.pendulumPosition"  # Set this to your serial port
serPendulum.baudrate = 9600

serCart.open()
serCart.flushInput()
serPendulum.open()
# serPendulum.flushInput()

# left 7-segment LED
cartLED_CLK = 23
cartLED_DIO = 24
cartLED = TM1637(cartLED_CLK, cartLED_DIO)

# right 7-segment LED
pendLED_CLK = 25
pendLED_DIO = 8
pendLED = TM1637(pendLED_CLK, pendLED_DIO)

GPIO_ENABLE_MOTOR_PIN = 38

##############
serAction = serial.Serial()
serAction.port = "/dev/ttyUSB.actionController"
serAction.baudrate = 9600
serAction.open()
serAction.flushInput()
serAction.flushOutput()
motorPowerDivisor = 2.5


##############


# Policy setup
class Policy(Enum):
    Gamepad = 1
    TFLite = 2  # formerly used for Fuzzy controller
    GamepadGP = 3
    BNN = 4
    # GamepadNFQ = 5
    PSO = 5  # formerly used for GamepadNFQ
    VQC = 6  # Quantum VQC Policy


currentPolicy = Policy.Gamepad


# Threads for receiving position of cart and velocity
class TPendulum(threading.Thread):
    def __init__(self):
        super(TPendulum, self).__init__(name="PendulumPos")

    def run(self):
        global pendulumPosition, cartPosition, UDP_IP, pendLED, cartLED
        # , pendLED, cartLED, currentPolicy
        oldMillis = int(round(time.time() * 1000))
        msDiffMovingAverage = 0
        alpha = 0.1
        serPendulum.flushInput()
        serPendulum.flushOutput()
        oldCartPos = 0
        oldPendPos = 0

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        while True:
            data = serPendulum.readline().decode("utf-8").replace("\n", "")
            if len(data) > 0:
                currentMillis = int(round(time.time() * 1000))
                serPendulum.flushInput()
                serPendulum.flushOutput()
                pendulumPosition = float(data)

                data = "%s;%s;%s;%s" % (cartPosition, (cartPosition - oldCartPos), pendulumPosition, (pendulumPosition - oldPendPos))
                oldCartPos = cartPosition
                oldPendPos = pendulumPosition

                sock.sendto(data.encode("utf-8"), (UDP_IP, 5000))  # Python: CartGP controller
                sock.sendto(data.encode("utf-8"), (UDP_IP, 6000))  # Python: ActionController Gamepad
                sock.sendto(data.encode("utf-8"), (UDP_IP, 7000))  # Java: BNN Policy
                sock.sendto(data.encode("utf-8"), (UDP_IP, 8000))  # Java: NFQ Policy

                timeDiff = currentMillis - oldMillis
                oldMillis = currentMillis
                msDiffMovingAverage = (1 - alpha) * msDiffMovingAverage + alpha * timeDiff
                # print ("Sent data: ", datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S.%f"), data, " msDiff=", timeDiff, " msDiffAvg=", msDiffMovingAverage)


class TCart(threading.Thread):
    def __init__(self):
        super(TCart, self).__init__(name="CartPos")

    def run(self):
        global cartPosition
        serCart.flushInput()
        serCart.flushOutput()
        while (True):
            data = serCart.readline().decode("utf-8").replace("\n", "")
            serCart.flushInput()
            serCart.flushOutput()
            cartPosition = float(data);


class TDisplay(threading.Thread):
    def __init__(self):
        super(TDisplay, self).__init__(name="7SegmentDisplay")

    def run(self):
        global cartPosition, pendulumPosition

        while True:
            sleep(0.05)
            # update 7-segement
            #show_CP_integer (cartLED, "C", 123)
            #show_CP_integer (pendLED, "P", 456)
            # '''
            c = "%s" % cartPosition
            p = "%s" % pendulumPosition
            if len(p) > 0:
                show_CP_integer(pendLED, "P", int(float(p)))
            else:
                show_CP_integer(pendLED, "P", 0)
            if len(c) > 0:
                show_CP_integer(cartLED, "C", int(float(c)))
            else:
                show_CP_integer(cartLED, "C", 0)
            # '''


class TActionController(threading.Thread):

    def __init__(self):
        super(TActionController, self).__init__(name="ActionController")
        global UDP_IP

    def run(self):

        global currentPolicy, motorPowerDivisor, gamepadAction, pendulumPosition, exploration

        ###########################################################
        # setup sockets with reduced Receive buffer size, in order
        # to avoid empying buffer with many "old" actions
        ###########################################################
        # FuzzyGP
        sockFGP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sockFGP.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
        sockFGP.bind((UDP_IP, 5001))
        # Gamepad
        sockGamepad = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sockGamepad.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
        sockGamepad.bind((UDP_IP, 6000))
        # BNN
        sockBNN = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sockBNN.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
        sockBNN.bind((UDP_IP, 7001))
        # NFQ
        # sockNFQ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # sockNFQ.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
        # sockNFQ.bind((UDP_IP, 8001))
        # Current Policy and Action
        sockPolicyAction = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        ################################
        # setup Python cart controllers
        ################################
        cgp = CartpoleGeneticProgrammingController()

        # #### MOOSE & BNN TF Policies ####
        # ctf = CartpoleTFLitePolicyController(tf_lite_model="../gpio/policies/moose_upswing.tflite", buffer_size=8)
        # cbnn = CartpoleTFLitePolicyController(tf_lite_model="../gpio/policies/moose_balancing_no_history.tflite", buffer_size=8, num_past_actions=3)
        # ctf = TestCartpoleTFLitePolicyController(tf_lite_model="../gpio/policies/test_models/bnn_policy.tflite", buffer_size=8)
        # cbnn = TestCartpoleBNNTFLitePolicyController(tf_lite_model="../gpio/policies/test_models/bnn_policy.tflite", buffer_size=8)
        cbnn = CartpoleTFLitePolicyController(tf_lite_model="../gpio/policies/moose_upswing_linux_6.tflite", buffer_size=10)

        # #### PSO Policies ####
        # cpso = CartpolePSOPolicyController(pso_weights_path="../gpio/policies/PSONN_balancing_action_with_cost", buffer_size=10)
        cpso = CartpoleTFLiteBSFController(tf_lite_model="../gpio/policies/BSF_25_9111.tflite", buffer_size=10)

        # #### Fuzzy Policies ####
        # cfgp = CartpoleFuzzyGP()
        # cfgp.setup(useSocket=True, udpPortState=5000, udpPortAction=5001)

        # #### Quantum VCQ Policies ####
        best_policy_path = "../gpio/policies/weights_input_output/checkpoint_000350"
        test_path = "../gpio/policies/test_models/20tm_3layers/checkpoint_000200"
        # cqu = CartpoleTFVQCPolicyController(tf_model="../gpio/policies/vqc/2-2-v1/vqc_model_weights", buffer_size=10)
        # cqu = CartpoleTFPolicyController(tf_model="../gpio/policies/final_checkpoints/classical/checkpoint_000150", buffer_size=8)
        # cqu = TestCartpoleTFCSupervisedPolicyController(tf_lite_model="../gpio/policies/best_weights_supervised_2.h5", buffer_size=8, num_past_actions=3)
        # cqu = TestCartpoleCLSupervisedPolicyController(tf_lite_model="../gpio/policies/real_cartpole_supervised.tflite", buffer_size=8, num_past_actions=3)
        # cqu = CartpoleTFVQCPolicyControllerTorch(tf_model="../gpio/policies/vqc/2-2-v2/model.pth", buffer_size=10)
        # cqu = CartpoleTFCircuitVQCPolicyController(tfc_model="../gpio/policies/vqc/10state-v1/tc_model_weights", buffer_size=10)
        cqu = TestCartpoleTFCPolicyController(tf_model=test_path, buffer_size=8, num_past_actions=3)

        ################
        # setup logging
        ################
        stateConverter = State()
        logFilename = ("/tmp/demonstration_classical_no_human_right_%s.csv" % datetime.now().strftime("%Y-%m-%d__%H-%M-%S"))

        csvfile = open(logFilename, 'w')
        fieldNames = ['datetime', 'cartPos', 'cartVel', 'pendPos', 'pendVel', 'action', 'cartController', 'exploration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldNames, delimiter=";")
        writer.writeheader()

        while True:

            action = 0
            explorationAction = 0
            dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            # print("\n%s" % dt)

            ##########
            # Fuzzy (deprecated)
            ##########
            # elif currentPolicy == Policy.Fuzzy:
            # 	print("Waiting for TFLite action")
            # 	action, controllerName, stateString = cfgp.getAction()
            # 	# stateString = "%s;%s;%s;%s" %  (state[0], state[1], state[2], state[3])
            #
            # 	if action == "NaN":
            # 		print("Received TFLite action: NaN, controllerName=%s, stateString=%s" % (controllerName, stateString))
            # 	else:
            # 		print("Received TFLite action: %s, controllerName=%s, stateString=%s" % (action, controllerName, stateString))

            ##########
            # Gamepad
            ##########
            if currentPolicy == Policy.Gamepad:
                #print("Waiting for Gamepad state")
                data, addr = sockGamepad.recvfrom(1024)
                data = data.decode("utf-8")
                stateString = data

                if buttonStatus[TGamepad.SCR_L]:
                    action = -gamepadAction

                elif buttonStatus[TGamepad.SCR_R]:
                    action = gamepadAction

                # print("Policy: Gamepad, action=%s" % action)

            ##################################################################################
            # Gamepad + GeneticProgramming if pendulumPosition < 25 || pendulumPosition > 335
            ##################################################################################
            elif currentPolicy == Policy.GamepadGP:
                #print("Waiting for GamepadGP state")
                data, addr = sockGamepad.recvfrom(1024)
                data = data.decode("utf-8")
                stateString = data
                controller = "Gamepad"
                # state = stateConverter.convertStringState(data)
                # cPos, pPos, cVel, pVel = state
                print("pendulumPosition=%s" % pendulumPosition)
                pPos = float(pendulumPosition)

                if buttonStatus[TGamepad.SCR_L]:
                    action = -gamepadAction

                elif buttonStatus[TGamepad.SCR_R]:
                    action = gamepadAction

                if pPos < 25 or pPos > 335:
                    # state = data.replace ("S", "").replace ("E", "").split (";")
                    state = stateConverter.convertStringState(data)
                    action = cgp(state)
                    controller = "GP"

                print("Policy: %s, action=%s" % (controller, action))

            ##################################################################################
            # TFLite Policy (all angular positions, i.e. including upswing)
            ##################################################################################
            elif currentPolicy == Policy.BNN:
                #print("Waiting for TFLite state")
                data, addr = sockGamepad.recvfrom(1024)
                data = data.decode("utf-8")
                stateString = data
                # state = stateConverter.convertStringState(data)
                # cPos, pPos, cVel, pVel = state
                print("pendulumPosition=%s" % pendulumPosition)
                pPos = float(pendulumPosition)

                if buttonStatus[TGamepad.SCR_L]:
                    action = -gamepadAction

                elif buttonStatus[TGamepad.SCR_R]:
                    action = gamepadAction

                controller = "BNN"
                state = stateConverter.convertStringState(data)

                # if pPos < 25 or pPos > 335:
                if pPos <= 180 or pPos > 180:
                    # state = data.replace ("S", "").replace ("E", "").split (";")
                    action = cbnn(state)

                print("Policy: %s, action=%s" % (controller, action))

            ##################################################################################
            # GamePad + Quantum VCQ Policy f pendulumPosition < 25 || pendulumPosition > 335
            ##################################################################################
            elif currentPolicy == Policy.VQC:
                #print("Waiting for TFLite state")
                data, addr = sockGamepad.recvfrom(1024)
                data = data.decode("utf-8")
                stateString = data
                # state = stateConverter.convertStringState(data)
                # cPos, pPos, cVel, pVel = state
                print("pendulumPosition=%s" % pendulumPosition)
                pPos = float(pendulumPosition)

                if buttonStatus[TGamepad.SCR_L]:
                    action = -gamepadAction

                elif buttonStatus[TGamepad.SCR_R]:
                    action = gamepadAction

                controller = "VQC"
                state = stateConverter.convertStringState(data)

                if pPos < 25 or pPos > 335:
                    # state = data.replace ("S", "").replace ("E", "").split (";")
                    action = cqu(state)

                print("Policy: %s, action=%s" % (controller, action))

            ##################################################################################
            # Gamepad + PSO Policy if pendulumPosition < 25 || pendulumPosition > 335
            ##################################################################################
            elif currentPolicy == Policy.PSO:
                #print("Waiting for PSO state")
                data, addr = sockGamepad.recvfrom(1024)
                data = data.decode("utf-8")
                stateString = data
                controller = "Gamepad"
                # state = stateConverter.convertStringState(data)
                # cPos, pPos, cVel, pVel = state
                print("pendulumPosition=%s" % pendulumPosition)
                pPos = float(pendulumPosition)

                if buttonStatus[TGamepad.SCR_L]:
                    action = -gamepadAction

                elif buttonStatus[TGamepad.SCR_R]:
                    action = gamepadAction

                if pPos < 25 or pPos > 335:
                    # state = data.replace ("S", "").replace ("E", "").split (";")
                    state = stateConverter.convertStringState(data)
                    action = cpso(state)
                    controller = "PSO"

                    print("Policy: %s, action=%s" % (controller, action))

            #######################
            # check for exploration
            #######################
            if random() < exploration:
                action = (random() * 2.) - 1.0
                explorationAction = 1
                print("Exploration action: %.3f" % action)

            ##############################
            # send Policy Action to socket
            ##############################
            sockPolicyAction.sendto(f"{action};{currentPolicy.value}".encode("utf-8"), (UDP_IP, 9999))

            #############################
            # Perform action on CartPole
            #############################
            iAction = int((action / motorPowerDivisor) * 63.) + 64
            # bAction = bytearray([int((action/motorPowerDivisor) * 63.)+64])
            bAction = bytearray([iAction])
            # print("performing action=%s (explorationRate=%.2f), sAction=%s, iAction=%d" % (action, exploration, chr(iAction), iAction))
            serAction.write(bAction)

            #################
            # Log transition
            #################
            cPos, cVel, pPos, pVel = stateString.split(";")
            fieldNames = ['datetime', 'cartPos', 'cartVel', 'pendPos', 'pendVel', 'action', 'cartController', 'exploration']
            writer.writerow({
                'datetime': dt,
                'cartPos': cPos,
                'cartVel': cVel,
                'pendPos': pPos,
                'pendVel': pVel,
                'action': action,
                'cartController': currentPolicy.name,
                'exploration': explorationAction
            })


slowActions = False
gamepadAction = 0.7
balanceAngle = 25
exploration = 0

# send positions 
UDP_IP = "127.0.0.1"

tCart = TCart()
tCart.start()
tPendulum = TPendulum()
tPendulum.start()
tGamepad = TGamepad()
tGamepad.start()
tDisplay = TDisplay()
tDisplay.start()
tAction = TActionController()
tAction.start()

oldButtonStatus = {}
while True:

    if not operator.eq(oldButtonStatus, buttonStatus):

        oldButtonStatus = dict(buttonStatus)

        if buttonStatus[TGamepad.SELECT] == 1 and buttonStatus[TGamepad.A] == 1:
            # Toggle fast/slow action flag in case SHIFT+A is pressed multiple times in sequence
            if currentPolicy == Policy.Gamepad:
                print("Selecting policy 'Gamepad'. slowActions=%s" % slowActions)
                if slowActions:
                    gamepadAction = 0.5
                else:
                    gamepadAction = 0.7
                slowActions = not slowActions
            else:
                print("Selecting policy 'Gamepad'. slowActions=%s" % slowActions)
                slowActions = False

            currentPolicy = Policy.Gamepad

        # ## SELECT+START => TFLite (Gamepad + TFLite Policy)
        # elif buttonStatus[TGamepad.SELECT] == 1 and buttonStatus[TGamepad.START] == 1:
        #     print("Selecting policy 'TFLite'")
        #     currentPolicy = Policy.TFLite

        ## SELECT+B => GamepadGP (Gamepad + Genetic Programming)
        elif buttonStatus[TGamepad.SELECT] == 1 and buttonStatus[TGamepad.B] == 1:
            print("Selecting policy 'GamepadGP'")
            currentPolicy = Policy.GamepadGP

        ## SELECT+X => BNN Policy (Bayesian Neural Network)
        elif buttonStatus[TGamepad.SELECT] == 1 and buttonStatus[TGamepad.X] == 1:
            print("Selecting policy 'BNN'")
            currentPolicy = Policy.BNN

        ## SELECT+Y => Quantum VCQ Policy
        elif buttonStatus[TGamepad.SELECT] == 1 and buttonStatus[TGamepad.Y] == 1:
            print("Selecting policy 'VQC'")
            currentPolicy = Policy.VQC

        ## Calibrate pendulum in upright position
        elif buttonStatus[TGamepad.SELECT] == 1 and buttonStatus[TGamepad.YAXIS] == 0:
            print("Calibrating pendulum in upright position")
            serPendulum.write("1".encode("utf-8"))

        ## Calibrate pendulum in down position
        elif buttonStatus[TGamepad.SELECT] == 1 and buttonStatus[TGamepad.YAXIS] == 255:
            print("Calibrating pendulum in down position")
            serPendulum.write("2".encode("utf-8"))

        ## decrease exploration rate by 1 %
        elif buttonStatus[TGamepad.SELECT] == 1 and buttonStatus[TGamepad.XAXIS] == 0:
            exploration = max(0, exploration - 0.01)
            print("decreasing exploration rate to %s" % exploration)

        ## increase exploration rate by 1 %
        elif buttonStatus[TGamepad.SELECT] == 1 and buttonStatus[TGamepad.XAXIS] == 255:
            exploration = min(1, exploration + 0.01)
            print("increasing exploration rate to %s" % exploration)

    sleep(0.1)
