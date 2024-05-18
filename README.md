项目名称：使用人工智能预测学生成绩

数据集信息：

该数据收集了两所葡萄牙学校的中学学生的学习成绩。数据属性包括学生成绩，人口统计，社会和与学校相关的特征），并通过使用学校报告和调查表进行收集。提供了两个有关两个不同学科表现的数据集：数学（mat）和葡萄牙语（por）。在[Cortez and Silva，2008]中，两个数据集是在二进制/五级分类和回归任务下建模的。重要说明：目标属性G3与属性G2和G1具有很强的相关性。发生这种情况是因为G3是最后的一年级（在第3阶段发布），而G1和G2分别对应第1和第2阶段等级。没有G2和G1的情况下预测G3更加困难，但是这种预测更为有用（有关更多详细信息，请参见纸本资料）。

属性信息：

school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira).
1.学校-学生学校（二进制：“ GP”-加布里埃尔·佩雷拉（Gabriel Pereira）或“ MS”-Mousinho da Silveira）

sex - student's sex (binary: 'F' - female or 'M' - male).
2.性别-学生的性别（二进制：“ F”-女性或“ M”-男性）

age - student's age (numeric: from 15 to 22).
3.年龄-学生的年龄（数字：15至22）

address - student's home address type (binary: 'U' - urban or 'R' - rural).
4.地址-学生的家庭住址类型（二进制：“ U”-城市或“ R”-农村）

famsize - family size (binary: 'LE3' - less than or equal to 3 or 'GT3' - greater than 3).
5.famsize-家庭大小（二进制：“ LE3”-小于或等于3或“ GT3”-大于3）

Pstatus- parent's cohabitation status (binary: 'T' - living together or 'A' - apart).
6.Pstatus-父母的同居状态（二进制：“ T”-同居或“ A”-分开）

Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education).
7.Medu-母亲的教育（数字：0-无，1-初等教育（四年级），2 – 5至9年级，3 –中等教育或4 –高等教育）

Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education).
8.Fedu-父亲的教育（数字：0-无，1-初等教育（四年级），2 – 5至9年级，3 –中等教育或4 –高等教育）

Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other').
9.Mjob-母亲的工作（名义：“教师”，“与健康”有关的，民事“服务”（例如行政或警察），“在家”或“其他”）

Fjob- father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other').
10.Fjob-父亲的工作（名义：“教师”，“与健康”相关的，民事“服务”（例如行政或警察），“在家”或“其他”）

reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other').
11.理由-选择这所学校的理由（名义：接近“家”，学校“声誉”，“课程”偏好或“其他”）

guardian - student's guardian (nominal: 'mother', 'father' or 'other').
12.监护人-学生的监护人（名词：“母亲”，“父亲”或“其他”）

traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour).
13.traveltime-学校到学校的旅行时间（数字：1-<15分钟，2-15至30分钟，3-30分钟至1小时或4-> 1小时）

studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours).
14.学习时间-每周学习时间（数字：1-<2小时，2-2至5小时，3-5至10小时或4-> 10小时）

failures - number of past class failures (numeric: n if 0 <= n < 3, else 3).
15.失败-过去类失败的次数（（数字：如果1 <= n <3，则为n，否则为3））

schoolsup - extra educational support (binary: yes or no).
16.schoolup-额外的教育支持（二进制：是或否）

famsup - family educational support (binary: yes or no).
17.famsup-家庭教育支持（二进制：是或否）

paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no).
18.付费-课程主题内的额外付费课程（数学或葡萄牙语）（二进制：是或否）

activities - extra-curricular activities (binary: yes or no).
19.活动-课外活动（二进制：是或否）

nursery - attended nursery school (binary: yes or no).
20.托儿所-上托儿所（二进制：是或否）

higher - wants to take higher education (binary: yes or no).
21.更高-想要接受高等教育（二进制：是或否）

internet - Internet access at home (binary: yes or no).
22.互联网-在家上网（二进制：是或否）

romantic - with a romantic relationship (binary: yes or no).
23.浪漫-具有浪漫关系（二进制：是或否）

famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent).
24.家族-家庭关系的质量（数字：从1-非常差到5-极好）

freetime - free time after school (numeric: from 1 - very low to 5 - very high).
25.空闲时间-放学后的空闲时间（数字：从1-非常低到5-非常高）

goout - going out with friends (numeric: from 1 - very low to 5 - very high).
26.外出-与朋友外出（数字：从1-非常低到5-非常高）

Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high).
27.Dalc-工作日酒精消耗（数字：从1-非常低到5-非常高）

Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high).
28.Walc-周末酒精消耗（数字：从1-非常低至5-非常高）

health - current health status (numeric: from 1 - very bad to 5 - very good).
29.健康-当前的健康状况（数字：从1-非常差到5-非常好）

absences - number of school absences (numeric: from 0 to 93).
30.缺勤-缺勤人数（数字：0到93）

Grades which are related with the course subject:

这些成绩与课程主题（数学或葡萄牙语）相关：

G1 - first period grade (numeric: from 0 to 20).
31.G1-第一期成绩（数字：0至20）

G2 - second period grade (numeric: from 0 to 20)
32.G2-第二学期成绩（数字：0至20）

G3 - final grade (numeric: from 0 to 20, Output Target)
33.G3-最终成绩（数字：0到20，输出目标）