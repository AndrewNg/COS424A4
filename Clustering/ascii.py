nonascii = bytearray(range(0x80, 0x100))
with open('../data/cluster9.txt','rb') as infile, open('../data/cluster9_parsed.txt','wb') as outfile:
    for line in infile: # b'\n'-separated lines (Linux, OSX, Windows)
        outfile.write(line.translate(None, nonascii))