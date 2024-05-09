import math
import numpy as np
from time import time

def fread(fileID, sizeA, precision, skip, mode):
    import struct
    array = np.zeros(sizeA, np.int32)
    prec = ''
    if precision == 'uint32' and mode == 'le':
        prec = '<i' # '<': little endian, 'i': int 4 bytes

    for i in range(sizeA):
        buf = fileID.read(skip)
        if not buf:
            break
        value = struct.unpack(prec, buf)[0]
        array[i] = value
        fileID.seek(fileID.tell() + skip)
        sizeA -= 1
    return array


def bitshift(dec_numb_array, shift):
    array = np.array(dec_numb_array)
    for i, dec_numb in enumerate(dec_numb_array):
        bin_num = bin(dec_numb)[2:]
        if(shift < 0):
            bin_num = bin_num[0:shift]
        else:
            bin_num = bin_num + '0'*shift
        try:
            array[i] = int(bin_num, 2)
        except ValueError:
            array[i] = 0
    return array


'''
# Codice Becattini
def fun(dec_numb, shift):
    bin_num = bin(dec_numb)[2:]
    if (shift < 0):
        bin_num = bin_num[0:shift]
    else:
        bin_num = bin_num + '0' * shift
    try:
        a = int(bin_num, 2)
    except ValueError:
        a = 0
    return a

def bitshift2(dec_numb_array, shift):
    array = np.fromiter(map(lambda x: fun(x, shift), dec_numb_array), np.int)
    return array
'''


def load_atis_data(filename, flipX=0, flipY=0):
    from collections import OrderedDict

    td_data = {'ts': [], 'x': [], 'y': [], 'p': []}

    f = open(filename, 'rb')

    header = OrderedDict()
    endOfHeader = 0
    numCommentLine = 0
    # tt = time()
    while (endOfHeader == 0):
        bod = f.tell() # ritorna la posizione del puntatore corrente
        tline = f.readline()
        tline = tline.decode('ISO-8859-1') # Necessario con Python 3.7
        #tline = f.read(32)

        if (tline[0] is not '%'):
            endOfHeader = 1
        else:
            words = tline.split()
            if (len(words) > 2):
                if (words[1] == 'Date'):
                    if (len(words) > 3):
                        header.update({words[1]: words[2] + ' ' + words[3]})
                else:
                    header.update({words[1]: words[2]})

            numCommentLine = numCommentLine+1

    # print(header)

    f.seek(bod)

    evType = 0
    evSize = 0
    if (numCommentLine>0):
        evType = ord(f.read(1))
        evSize= ord(f.read(1))
    # print('evType:', evType, 'evSize:', evSize)


    bof = f.tell()
    f.seek(0, 2)
    numEvents = math.floor((f.tell()-bof)/evSize)
    #numEvents = 6000000
    #numEvents = 5
    # print('Numero Totale di Eventi Rilevati:', numEvents)

    # t1 = time()
    f.seek(bof)
    allTs=fread(fileID=f, sizeA=numEvents, precision='uint32', skip=4, mode='le')
    # print('t1: {}'.format(time() - t1))
    # t2 = time()
    f.seek(bof+4)
    allAddr=fread(fileID=f, sizeA=numEvents, precision='uint32', skip=4, mode='le')
    # print('t2: {}'.format(time() - t2))

    f.close()

    td_data['ts'] = np.double(allTs)

    version = 0
    if ('Version' in header.keys()):
        version = int(header['Version'])

    if (version) < 2:
        xmask = int('000001FF', 16)
        ymask = int('0001FE00', 16)
        polmask = int('00020000', 16)
        xshift = 0
        yshift = 9
        polshift = 17
    else:
        xmask = int('00003FFF', 16)
        ymask = int('0FFFC000', 16)
        polmask = int('10000000', 16)
        xshift = 0
        yshift = 14
        polshift = 28

    addr = abs(allAddr)
    td_data['x'] = np.double(bitshift((addr & xmask), -xshift))
    td_data['y'] = np.double(bitshift((addr & ymask), -yshift))
    td_data['p'] = -1+2*np.double(bitshift((addr & polmask), -polshift))

    '''
        print(' ')
        print('x_vect:', td_data['x'])
        print('y_vect:', td_data['y'])
        print('pol_vect:', td_data['p'])
        print('ts_vect', td_data['ts'])
    '''

    if (flipX > 0):
        td_data['x'] = flipX - td_data['x']

    if (flipY > 0):
        td_data['y'] = flipX - td_data['y']

    # return td_data
    return td_data['x'], td_data['y'], td_data['ts'], td_data['p']


if __name__ == "__main__":
    load_atis_data('log_td.dat')
	
