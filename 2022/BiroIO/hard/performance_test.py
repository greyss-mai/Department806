import subprocess

CMD = 'build/Lab_1'

N_START = 32
N_END   = 1024 * 2
N_STEP  = 64

file = open('performance_test_result.csv', 'w')
file.write('NM;CPU_TIME;GPU_TIME\n')

def parse_output(bytes):
    string = str(bytes)[2:].replace("'", '').split("\\n")[:-1]
    results = [
        int(string[0].replace('N: ', '')) * int(string[1].replace('M: ', '')), 
        int(string[2].replace('CPU_time: ', '').replace(' microseconds.', '')), 
        int(string[3].replace('GPU_time: ', '').replace(' microseconds.', ''))
    ]
    return results

table_frame_format = '{};{};{}\n'

for N in range(N_START, N_END, N_STEP):
    output = subprocess.check_output([CMD, str(N), str(N)])
    test_result = parse_output(output)
    print(test_result)
    file.write(table_frame_format.format(test_result[0], test_result[1], test_result[2]))

file.close()
