import os
def create_testbench(labels, bus_size, control_signals = []):
    tb = "VDD VDD 0 DC 5\n"

    count = 0
    for idx,cnt_sig in enumerate(control_signals):
        tb += f"V{cnt_sig.upper()} {cnt_sig.upper()} 0 pulse(0 5 {10 * ((2 ** (idx + 1)))}ms 1ns 1ns {10 * ((2 ** (idx + 1)))}ms {20 * ((2 ** (idx + 1)))}ms)\n"
        count+= 1

    for idx,label in enumerate(labels):

        for bus in range(bus_size):
            tb += f"V{label.upper()}{bus} {label.upper()}{bus} 0 pulse(0 5 {10*((2**(idx+count+1)))}ms 1ns 1ns {10*((2**(idx+count+1)))}ms {20*((2**(idx+count+1)))}ms)\n"


    tb+= f".tran {(2**(len(labels)+count))*20}ms"
    return tb


def ayse_ALU():
    tb = create_testbench(['a','b'], 16, ['t','z','y','x'])

    save_path = os.path.join("testbenches",'ALU_TB.txt')


    with open(save_path,"w") as file:
        file.write(tb)

def adder_TB():
    tb = create_testbench(['a', 'b'], 16, ['cin'])

    save_path = os.path.join("testbenches", 'ADDER_TB.txt')

    with open(save_path, "w") as file:
        file.write(tb)

def HalfAdder_TB():
    tb = create_testbench(['a', 'b'], 16, [])

    save_path = os.path.join("testbenches", 'HALFADDER_TB.txt')

    with open(save_path, "w") as file:
        file.write(tb)


def CarryGenerator_TB():
    tb = create_testbench(['p', 'g'], 4, ['c0'])

    save_path = os.path.join("testbenches", 'CarryGenerator_TB.txt')

    with open(save_path, "w") as file:
        file.write(tb)

def LogicUnit_TB():
    tb = create_testbench(['a', 'b'], 16, ['s0','s1'])

    save_path = os.path.join("testbenches", 'LogicUnit_TB.txt')

    with open(save_path, "w") as file:
        file.write(tb)


LogicUnit_TB()