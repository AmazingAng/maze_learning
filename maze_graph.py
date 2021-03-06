maze1_graph = {1: [2, 13],
         2: [1],
         3: [4, 15],
         4: [3, 5],
         5: [4, 6],
         6: [5, 7, 18],
         7: [6, 8],
         8: [7],
         9: [10, 21],
         10: [9, 11],
         11: [10, 12],
         12: [11, 24],
         13: [1, 14, 25],
         14: [13, 26],
         15: [3, 27],
         16: [17, 28],
         17: [16, 18, 29],
         18: [6, 17],
         19: [20, 31],
         20: [19, 21],
         21: [9, 20],
         22: [23, 34],
         23: [22, 24],
         24: [12, 23,36],
         25: [13],
         26: [14,27],
         27: [26, 15],
         28: [16],
         29: [17, 30],
         30: [29, 31, 42],
         31: [30, 19],
         32: [33, 44],
         33: [32, 34],
         34: [22, 33],
         35: [36],
         36: [35, 24],
         37: [38, 49],
         38: [37, 39],
         39: [40, 38, 51],
         40: [39],
         41: [42],
         42: [41, 30],
         43: [55],
         44: [32, 45],
         45: [44, 46],
         46: [45, 47],
         47: [46, 48],
         48: [47, 60],
         49: [37, 61],
         50: [51, 62],
         51: [39, 50,52],
         52: [51],
         53: [54],
         54: [53, 55,66],
         55: [43, 54,67],
         56: [57, 68],
         57: [56, 58],
         58: [57, 59],
         59: [58, 60],
         60: [48, 59],
         61: [49, 73],
         62: [50, 74],
         63: [64, 75],
         64: [63, 65],
         65: [64, 66],
         66: [54, 65],
         67: [55, 79],
         68: [56, 69],
         69: [68, 70],
         70: [69, 71],
         71: [70, 72],
         72: [71, 84],
         73: [61, 85],
         74: [62, 75],
         75: [74, 63],
         76: [77, 88],
         77: [76, 89],
         78: [79, 90],
         79: [78, 67],
         80: [81, 92],
         81: [80, 82],
         82: [81, 94],
         83: [95, 84],
         84: [72, 83, 96],
         85: [73, 97],
         86: [98],
         87: [88, 99],
         88: [87, 76],
         89: [77, 101],
         90: [78, 91],
         91: [90, 103],
         92: [80, 104],
         93: [105],
         94: [82, 95,106],
         95: [83, 94],
         96: [84],
         97: [85, 98, 109],
         98: [97, 86],
         99: [87, 100],
         100: [99, 112],
         101: [89, 102],
         102: [101, 114],
         103: [91, 104],
         104: [103, 92],
         105: [93, 106],
         106: [105, 94],
         107: [108, 119],
         108: [107, 120],
         109: [97, 110,121],
         110: [109, 122],
         111: [112, 123],
         112: [100, 111],
         113: [114, 125],
         114: [113, 102],
         115: [116, 127],
         116: [115, 117],
         117: [116, 129],
         118: [119, 130],
         119: [118, 107],
         120: [108],
         121: [109, 133],
         122: [110, 123],
         123: [111, 122],
         124: [125, 136],
         125: [124, 113],
         126: [127, 138],
         127: [115, 126],
         128: [129, 140],
         129: [128, 117,141],
         130: [118, 131,142],
         131: [130, 132],
         132: [131, 144],
         133: [121, 134],
         134: [133, 135],
         135: [134],
         136: [124, 137],
         137: [136, 138],
         138: [137, 126],
         139: [140],
         140: [139, 128],
         141: [129, 142],
         142: [130, 141,143],
         143: [142],
         144: [132],
        }

maze2_graph = {1: [2],
         2: [1,3,14],
         3: [2,15],
         4: [16],
         5: [6, 17],
         6: [5, 7],
         7: [6, 8,19],
         8: [7,20],
         9: [10,21],
         10: [9, 11],
         11: [10, 12],
         12: [11, 24],
         13: [14, 25],
         14: [2,13],
         15: [3, 16],
         16: [4, 15],
         17: [5, 29],
         18: [19, 30],
         19: [7, 18],
         20: [8],
         21: [9, 33],
         22: [23, 34],
         23: [22, 24],
         24: [12, 23,36],
         25: [13,37],
         26: [27],
         27: [26, 39],
         28: [29,40],
         29: [17, 28],
         30: [18, 31],
         31: [30, 43],
         32: [33, 44],
         33: [21, 32],
         34: [22, 46],
         35: [36,47],
         36: [35, 24],
         37: [25, 49],
         38: [50],
         39: [27, 51],
         40: [28,52],
         41: [42,53],
         42: [41],
         43: [31,44],
         44: [32,43, 56],
         45: [46, 57],
         46: [34, 45],
         47: [35],
         48: [ 60],
         49: [37, 61],
         50: [38, 51,62],
         51: [39, 50,52],
         52: [51,40,53],
         53: [41,52,65],
         54: [55,66],
         55: [54,67],
         56: [44],
         57: [45, 69],
         58: [70],
         59: [71, 60],
         60: [48, 59,72],
         61: [49, 73],
         62: [50],
         63: [64, 75],
         64: [63, 65],
         65: [64, 53],
         66: [54, 78],
         67: [55, 68],
         68: [67, 69],
         69: [68, 57],
         70: [58,82, 71],
         71: [70, 59],
         72: [60, 84],
         73: [61, 74],
         74: [73, 86],
         75: [76, 63],
         76: [75, 88],
         77: [78, 89],
         78: [77,66, 90],
         79: [80, 91],
         80: [79, 92],
         81: [82],
         82: [70,81, 94],
         83: [95, 84],
         84: [72, 83],
         85: [86, 97],
         86: [74,85,87],
         87: [86],
         88: [100, 76],
         89: [77, 101],
         90: [78],
         91: [79, 103],
         92: [80, 93],
         93: [92,105],
         94: [82,106],
         95: [83, 96,107],
         96: [95,108],
         97: [85, 98],
         98: [97, 110],
         99: [111, 100],
         100: [99, 88],
         101: [89, 102],
         102: [101, 114],
         103: [91, 104],
         104: [103, 116],
         105: [93, 106,117],
         106: [105, 94],
         107: [95],
         108: [96, 120],
         109: [110,121],
         110: [98,109, 122],
         111: [99, 123],
         112: [113, 124],
         113: [112],
         114: [115, 102],
         115: [116, 114],
         116: [115, 104],
         117: [105, 129],
         118: [119],
         119: [118, 120,131],
         120: [108,119],
         121: [109, 133],
         122: [110],
         123: [111, 135],
         124: [125, 112],
         125: [124, 126],
         126: [125, 138],
         127: [128, 139],
         128: [127, 140],
         129: [117,141],
         130: [131,142],
         131: [130, 119],
         132: [144],
         133: [121, 134],
         134: [133, 135],
         135: [123,134,136],
         136: [135, 137],
         137: [136, 138],
         138: [137, 126],
         139: [127],
         140: [141, 128],
         141: [129, 140],
         142: [130,143],
         143: [142,144],
         144: [132,143],
        }

CorrectPath_maze_1 = [1,
13,
14,
26,
27,
15,
3,
4,
5,
6,
18,
17,
29,
30,
31,
19,
20,
21,
9,
10,
11,
12,
24,
23,
22,
34,
33,
32,
44,
45,
46,
47,
48,
60,
59,
58,
57,
56,
68,
69,
70,
71,
72,
84,
83,
95,
94,
82,
81,
80,
92,
104,
103,
91,
90,
78,
79,
67,
55,
54,
66,
65,
64,
63,
75,
74,
62,
50,
51,
39,
38,
37,
49,
61,
73,
85,
97,
109,
110,
122,
123,
111,
112,
100,
99,
87,
88,
76,
77,
89,
101,
102,
114,
113,
125,
124,
136,
137,
138,
126,
127,
115,
116,
117,
129,
141,
142,
130,
131,
132,
144
]

CorrectPath_maze_2 = [1,
2,
14,
13,
25,
37,
49,
61,
73,
74,
86,
85,
97,
98,
110,
109,
121,
133,
134,
135,
123,
111,
99,
100,
88,
76,
75,
63,
64,
65,
53,
52,
40,
28,
89,
17,
5,
6,
7,
19,
18,
30,
31,
43,
44,
32,
33,
21,
9,
10,
11,
12,
24,
23,
22,
34,
46,
45,
57,
69,
68,
67,
55,
54,
66,
78,
77,
89,
101,
102,
114,
115,
116,
104,
103,
91,
79,
80,
92,
93,
105,
106,
94,
82,
70,
71,
59,
60,
72,
84,
83,
95,
96,
108,
120,
119,
131,
130,
142,
143,
144
]

IncorrectPath_maze_1 = [2,
                        25,
                        7,      8,
                        16,     28,
                        42,     41,
                        36,	35,
                        96,
                        106,	105,	93,
                        43,
                        53,
                        52,
                        40,
                        98,	86,
                        121,	133,	134,	135,
                        128,	140,	139,
                        143,
                        118,	119,	107,	108,	120]

separate_point1 = [0.5,
                   1.5,
                   3.5,
                   5.5,
                   7.5,
                   9.5,
                   10.5,
                   13.5,
                   14.5,
                   15.5,
                   16.5,
                   17.5,
                   19.5,
                   23.5,
                   26.5,
                   27.5]

IncorrectPath_maze_2 = [3,     15,     16,     4,
                        87,
                        122,
                        136,    137,    138,    126,    125,    124,    112,    113,
                        41,     42,
                        51,     39, 	27,     26,     50,     38,     62,
                        8,      20,
                        56,
                        36,     35,     47,
                        90,
                        117,    129,   	141,	140,	128,	127,	139,
                        81,
                        58,
                        48,
                        107,
                        118,
                        132]

separate_point2 = [3.5,
                 4.5,
                 5.5,
                 13.5,
                 15.5,
                 22.5,
                 24.5,
                 25.5,
                 28.5,
                 29.5,
                 36.5,
                 37.5,
                 38.5,
                 39.5,
                 40.5,
                 41.5]
