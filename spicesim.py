import matplotlib.pyplot as plt
import numpy as np
import os
from string import digits
import re

class Simulation:

    def __init__(self, filename, contol_signals = [], period_signal = 0):
        self.simulation_time = 0
        self.control_signals = contol_signals
        self.names, self.values, self.time, self.label_names, self.bus_nums = self.parse(filename)
        self.per_idx = None
        self.per_count = None
        self.per_idx_int = None
        if self.control_signals:
            self.find_period(self.control_signals[period_signal]) # 0
        else:
            self.find_period(self.names[0])

        self.period = self.simulation_time / self.per_count

        self.mid_per_times = [i*self.period + self.period/2 for i in range(self.per_count)]
        print(self.per_count,self.simulation_time,self.mid_per_times)

        self.text_coord = -(self.simulation_time/0.04)*(0.08/(16))

        self.filename = filename.split('.')[0]
        self.output_dir = 'output'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def parse(self, filename):

        with open(filename, 'r') as tb:
            TestBench = tb.readlines()

        line_array = []
        for line in TestBench:
            line_array.append(line.split('\t'))

        header = line_array[0]
        #print(header)
        header = [name.replace('\n','').split('V(')[1][:-1] for name in header[1:]]
        self.simulation_time = float(line_array[-1][0]) - float(line_array[1][0])
        period = self.simulation_time / (len(line_array) - 1)
        print("Simulation Time:", self.simulation_time * 1000, "ms")
        print("Period", period * 1000, "ms")
        print(header)
        data = np.array(line_array[1:], dtype=np.float)

        time = data[:, 0]
        values = {}
        label_names = {}
        bus_nums = {}
        table = str.maketrans('', '', digits)
        for i in range(len(header)):
            values[header[i]] = data[:, i + 1]

            if header[i] in self.control_signals:
                print(header[i])
                continue
            label_names[header[i].translate(table)] = None
            bus_nums[re.sub("[^0-9]","",header[i])] = None
        print("a",label_names.keys())
        return header, values, time, list(label_names.keys()), list(bus_nums.keys())


    def find_period(self,signal_name):

        signal = self.values[signal_name]
        boolean, _ = self.make_boolean_init(signal_name)

        x = boolean[0]

        per_count = 1
        per_idx = [self.time[0]]
        per_idx_int = [0]
        for idx, i in enumerate(boolean):
            if x != i:
                per_count += 1
                per_idx.append(self.time[idx])
                per_idx_int.append(idx)
                x = i

        self.per_idx = per_idx
        self.per_count = per_count
        self.per_idx_int = per_idx_int

    def make_boolean(self, signal_name):
        corrected = np.interp(self.mid_per_times, self.time, self.values[signal_name])
        filtered = np.round(corrected / 5)
        boolean = filtered.astype('bool')
        return boolean,1
        x = boolean[0]

        time_temp = self.time[0]

        index_temp = 0
        per_int_idx = [0]

        for idx, i in enumerate(boolean):

            if x != i:

                if self.time[idx] - time_temp < 3 / 1000:
                    boolean[index_temp - 1:idx + 1] = i
                    continue

                x = i
                time_temp = self.time[idx]
                index_temp = idx

        x = boolean[0]
        for idx, i in enumerate(boolean):
            if x != i:
                per_int_idx.append(idx)
                x = i

        return boolean, np.array(per_int_idx)

    def make_boolean_init(self, signal_name):
        #corrected = np.interp(self.mid_per_times, self.time, self.values[signal_name])
        filtered = np.round(self.values[signal_name] / 5)
        boolean = filtered.astype('bool')

        x = boolean[0]

        time_temp = self.time[0]

        index_temp = 0
        per_int_idx = [0]

        for idx, i in enumerate(boolean):

            if x != i:

                if self.time[idx] - time_temp < 3/1000:
                    boolean[index_temp-1:idx+1] = i
                    continue

                x = i
                time_temp = self.time[idx]
                index_temp = idx

        x = boolean[0]
        for idx,i in enumerate(boolean):
            if x != i:
                per_int_idx.append(idx)
                x = i

        return boolean, np.array(per_int_idx)

    def draw_one_signal_boolean(self, signal_name, order = 1, num_figure = 3 , sep = False):


        signal, idx = self.make_boolean(signal_name)
        signal=np.append(signal,signal[-1])

        time = self.mid_per_times.copy()

        time.append(self.mid_per_times[-1] + self.mid_per_times[0]*2)

        plt.step(np.array(time)-self.mid_per_times[0], signal + (num_figure-order)*2, 'r', linewidth = 2, where='post')
        plt.text(self.text_coord, (num_figure - order) * 2 + 0.5, str(signal_name).upper()[0]+'['+str(signal_name)[1:]+']', fontweight="bold",verticalalignment='center')

        #corrected = np.interp(self.mid_per_times,self.time,signal)

        for idx2,i in enumerate(signal[:-1]):
            #arr = abs(idx - i)
            #corrected_idx = idx[np.where(arr == min(arr))][0]
            plt.text(self.simulation_time/(self.per_count)*(idx2)+self.simulation_time/self.per_count/2,(num_figure-order)*2+0.5, str(int(i)), fontweight="bold", horizontalalignment='center',
     verticalalignment='center')

        return signal

    def draw_one_signal(self, signal_name, order = 1, num_figure = 3):
        signal_raw = np.round(self.values[signal_name]/5)
        signal, idx = self.make_boolean(signal_name)
        plt.plot(self.time, signal_raw + (num_figure-order)*2, 'r', linewidth = 2)
        plt.text(self.text_coord, (num_figure - order) * 2 + 0.5, str(signal_name).upper()[0]+'['+str(signal_name)[1:]+']', fontweight="bold",verticalalignment='center')

        for idx2,i in enumerate(self.per_idx_int):
            arr = abs(idx - i)
            corrected_idx = idx[np.where(arr == min(arr))][0]
            plt.text(self.simulation_time / (self.per_count) * (idx2) + self.simulation_time / self.per_count / 2,
                     (num_figure - order) * 2 + 0.5, str(int(signal[corrected_idx])), fontweight="bold",
                     horizontalalignment='center',
                     verticalalignment='center')

    def draw_one_pair_boolean(self,bus,save = False, sep = True):
        #print(self.label_names)
        figsize = plt.rcParams.get('figure.figsize')
        if sep:
            plt.figure(figsize=(figsize[0]*2, figsize[1]))

        self.draw_line('x', self.per_idx, color='.5', linewidth=2)
        count = 0
        cont_corrected = {}
        if self.control_signals:
            for idx, label in enumerate(self.control_signals):
                values = self.draw_one_signal_boolean(label, order=idx + 1, num_figure=len(self.label_names)+len(self.control_signals))
                cont_corrected[label] = values
                count += 1
        bus_corrected = {}
        for idx, label in enumerate(self.label_names):
            values = self.draw_one_signal_boolean(label + str(bus), order = idx+count+1, num_figure=len(self.label_names)+len(self.control_signals))
            bus_corrected[label] = values
        plt.xlim([self.text_coord,self.simulation_time])
        plt.gca().axis('off')

        if save:
            save_path = os.path.join(self.output_dir, self.filename +'_bus'+str(bus)+ '_out.png')
            plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)

        return cont_corrected, bus_corrected
    def draw_one_pair(self, bus, save = False):
        self.draw_line('x', self.per_idx, color='.5', linewidth=2)

        for idx, label in enumerate(self.label_names):
            self.draw_one_signal(label + str(bus), order = idx+1, num_figure=len(self.label_names))

        plt.xlim([self.left_coord, self.simulation_time])
        plt.gca().axis('off')

        if save:
            save_path = os.path.join(self.output_dir, self.filename + '_bus' + str(bus) + '_out_raw.png')
            plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)

    def draw_line(self, ax, pos, *args, **kwargs):
        if ax == 'x':
            for p in pos:
                plt.axvline(p, *args, **kwargs)
        else:
            for p in pos:
                plt.axhline(p, *args, **kwargs)

    def draw_all_boolean(self,num_col = 4, save = False, sep = False):

        max_num = len(self.bus_nums)
        row = max_num/num_col+1

        cont_all = {}
        den = [""]*self.per_count
        print(den)
        bus_all = dict.fromkeys(self.label_names, None)
        for label in bus_all:
            bus_all[label] = den.copy()
        figsize = plt.rcParams.get('figure.figsize')

        plt.figure(figsize=(figsize[0]*num_col,figsize[1]*row))
        for idx in range(max_num):
            plt.subplot(row,num_col,idx+1)
            cont, bus = self.draw_one_pair_boolean(idx, sep=sep)

            for label in bus.keys():
                for i in range(self.per_count):
                    bus_all[label][i] = str(int(bus[label][i])) + bus_all[label][i]
                    #print(label,bus[label])
                    #break
            if sep:
                save_path = os.path.join(self.output_dir, self.filename + '_out_' + str(idx) + '.png')
                plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)

        for label in cont:
            x = [str(int(num)) for num in cont[label]]
            res = ""
            for chr in x:
                res += chr
            cont[label] = res

        cont_all = cont
        #for label in bus_all:
        #    bus_all[label] = bus_all[label][-1]
        if save and not sep:
            save_path = os.path.join(self.output_dir,self.filename+'_out.png')
            plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)

        return cont_all, bus_all

    def draw_all(self,num_col = 4, save = False):
        max_num = len(self.bus_nums)
        row = max_num/num_col+1
        figsize = plt.rcParams.get('figure.figsize')

        plt.figure(figsize=(figsize[0]*4,figsize[1]*row))
        for idx in range(max_num):
            plt.subplot(row,num_col,idx+1)
            self.draw_one_pair(idx)

        if save:
            save_path = os.path.join(self.output_dir,self.filename+'_out_raw.png')
            plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)



"""
sim = Simulation("XOR2x1_16bit.txt",'a','b','f')
#sim.draw_one_signal_boolean('a0')
sim.draw_one_pair_boolean(15, save=True)
sim.draw_all_boolean(save = True)
#plt.show()

"""




def simulate_adder():
    sim = Simulation("CarryLookAheadAdder_16bit.txt", ['cin', 'cout'])
    cont, bus = sim.draw_all_boolean(save=True, sep=True)

    sim_time = len(bus['a'])

    cin_all = cont['cin']
    cout_all = cont['cout']
    a_all = bus['a']
    b_all = bus['b']
    f_all = bus['f']
    flag = True
    for i in range(sim_time):
        cin = cin_all[i]
        cout = cout_all[i]
        a = a_all[i]
        b = b_all[i]
        f = f_all[i]
        a_int = int(a, 2)
        b_int = int(b, 2)
        f_int = int(f, 2)
        cin_int = int(cin, 2)
        res = cout + f
        res_int = int(res,2)


        exp_int = a_int+b_int+cin_int
        exp = bin(exp_int)
        if res_int!=exp_int:

            flag = False
            print(f"Sim Failed Time = {i+1}  f={f} res={res} a={a} b={b} cin={cin} cout{cout} exp = {exp}")
            return -1

    print("Simulation finished without error")
    return True



def simulate_mux4x1_16bit():
    sim = Simulation("MUX4x1_16bit.txt", ['s0', 's1'])
    cont, bus = sim.draw_all_boolean(save=True, sep=True)

    sim_time = len(bus['a'])

    s0 = cont['s0']
    s1 = cont['s1']
    a = bus['a']
    b = bus['b']
    c = bus['c']
    d = bus['d']
    f = bus['f']
    flag = True
    for i in range(sim_time):

        sel = int(s1[i]+s0[i],2)
        if sel == 0:
            if f[i] != a[i]:
                flag = False
                print(f"Sim Failed Time = {i+1} Sel = {sel} f={f[i]} exp = {a[i]}")
                return -1
        if sel == 1:
            if f[i] != b[i]:
                flag = False
                print(f"Sim Failed Time = {i+1} Sel = {sel} f={f[i]} exp = {b[i]}")
                return -1

        if sel == 2:
            if f[i] != c[i]:
                flag = False
                print(f"Sim Failed Time = {i+1} Sel = {sel} f={f[i]} exp = {c[i]}")
                return -1

        if sel == 3:
            if f[i] != d[i]:
                flag = False
                print(f"Sim Failed Time = {i+1} Sel = {sel} f={f[i]} exp = {d[i]}")
                return -1

    print("Simulation finished without error")
    return True


def simulate_alu_ayse():
    sim = Simulation("ALUfinal.txt", ['cout', 'x', 'y', 'z', 't', 'f0', 'f1','vdd'], period_signal=4)
    cont, bus = sim.draw_all_boolean(save=False, sep=True)

    sim_time = len(bus['a'])
    print("SIM TIME = ",sim_time)
    cout = cont['cout']
    x = cont['x']
    y = cont['y']
    z = cont['z']
    t = cont['t']
    f0 = cont['f0']
    f1 = cont['f1']

    a_all = bus['a']
    b_all = bus['b']
    o_all = bus['o']
    flag = True
    for i in range(sim_time):
        opcode = int(x[i]+y[i]+z[i]+t[i],2)
        comp = int(f1[i]+f0[i],2)
        a = a_all[i]
        b = b_all[i]
        o = o_all[i]
        a_int = int(a,2)
        b_int = int(b,2)
        o_int = int(o,2)

        if opcode == 0:
            if o_int != (65535-a_int):
                print(f"Error OPCODE {opcode} at timestep {i}")
                return False

        elif opcode == 1:
            if o_int != (a_int & b_int):
                print(f"Error OPCODE {opcode} at timestep {i}")
                return False

        elif opcode == 2:
            if o_int != (a_int | b_int):
                print(f"Error OPCODE {opcode} at timestep {i}")
                return False

        elif opcode == 3:
            if o_int != (a_int ^ b_int):
                print(f"Error OPCODE {opcode} at timestep {i}")
                return False

        elif opcode == 4:
            res = a_int + b_int
            if res > 65535:
                res -= 65536

            if o_int != res:
                print(a,b,o)
                print(f"Error OPCODE {opcode} at timestep {i}")
                return False

        elif opcode == 5:
            if a_int>=b_int:
                if o_int != (a_int - b_int):
                    print(f"Error OPCODE {opcode} at timestep {i}")
                    return False

            if a_int<b_int:
                res = (a_int + ~b_int +1) & 0b1111111111111111
                print(a,b,o,res)
                if o_int != res:
                    print(f"Error OPCODE {opcode} at timestep {i}")
                    return False

        elif opcode == 6 or opcode == 7 or opcode == 12:
            if a_int>b_int:
                if comp != 0:
                    print(f"Error OPCODE {opcode} at timestep {i}")
                    return False
            elif a_int<b_int:
                if comp != 1:
                    print(f"Error OPCODE {opcode} at timestep {i}")
                    return False
            else:
                if comp != 2:
                    print(f"Error OPCODE {opcode} at timestep {i}")
                    return False

        elif opcode == 8:
            if o_int != int(a[1:]+"0",2):
                print(f"Error OPCODE {opcode} at timestep {i}")
                return False

        elif opcode == 9:
            if o_int != int("0"+a[0:-1],2):
                print(f"Error OPCODE {opcode} at timestep {i}")
                return False

        elif opcode == 10:
            if o_int != int(a[1:]+a[0],2):
                print(f"Error OPCODE {opcode} at timestep {i}")
                return False

        elif opcode == 11:
            if o_int != int(a[-1]+a[0:-1],2):
                print(f"Error OPCODE {opcode} at timestep {i}")
                return False


    print("Simulation finished without error")
    return True


def simulate_logicunit():
    sim = Simulation("LogicUnit.txt", ['s0','s1'])
    cont, bus = sim.draw_all_boolean(save=False, sep=True)

    sim_time = len(bus['a'])
    print("SIM TIME = ",sim_time)

    s0_all = cont['s0']
    s1_all = cont['s1']


    a_all = bus['a']
    b_all = bus['b']
    f_all = bus['f']
    flag = True
    for i in range(sim_time):
        s0 = s0_all[i]
        s1 = s1_all[i]

        opcode = int(s1+s0,2)

        a = a_all[i]
        b = b_all[i]
        f = f_all[i]
        a_int = int(a,2)
        b_int = int(b,2)
        f_int = int(f,2)

        if opcode == 0:
            if f_int != (a_int & b_int):
                print(f"Sim Failed Time = {i + 1}  f={f} op = {opcode} res={(a_int & b_int)} a={a} b={b}")
                return -1

        elif opcode == 1:
            if f_int != (a_int | b_int):
                print(f"Sim Failed Time = {i + 1}  f={f} op = {opcode} res={(a_int | b_int)} a={a} b={b}")
                return -1

        elif opcode == 2:
            if f_int != (~(a_int ^ b_int)) & 0b1111111111111111:
                print(f"Sim Failed Time = {i + 1}  f={f} op = {opcode} res={(a_int | b_int)} a={a} b={b}")

                return False

        elif opcode == 3:
            if f_int != (~(a_int)) & 0b1111111111111111:
                print(f"Sim Failed Time = {i + 1}  f={f} op = {opcode} res={(a_int | b_int)} a={a} b={b}")

                return False



    print("Simulation finished without error")
    return True


#simulate_alu_ayse()

#simulate_adder()

#simulate_mux4x1_16bit()
simulate_logicunit()