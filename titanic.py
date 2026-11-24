import pandas as pd
import matplotlib.pyplot as plt
try:
    df = pd.read_csv('titanic.csv')
    print('Dataset Found.')
except FileNotFoundError:
    print('File Not Found.')
    exit()
#Calculation Part (Pandas)
print('Rows and Columns: ', df.shape)
print('First 5 records: ', df.head())
print('Missing values: ', df.isnull().sum())

survival_rate = df['Survived'].mean() * 100
print('Survival Rate: ', survival_rate.round(1))


survival_gender = df.groupby('Sex')['Survived'].mean() * 100
print('Survival Rate by gender: ', survival_gender.round(1))


survival_ticket = df.groupby('Pclass')['Survived'].mean() * 100
print('Survival Rate by ticket: ', survival_ticket.round(1))


children = df[df['Age'] < 18]
adults = df[df['Age'] >= 18]
children_survived = children['Survived'].mean() * 100
adults_survived = adults['Survived'].mean() * 100

print('Children survival rate: ', children_survived.round(1))
print('Adult survival rate: ', adults_survived.round(1))

#Visualizatioin Part (Matplotlib)
#Survival Rate by Pclass
plt.figure('Survival Rate 1', figsize=( 7, 5))
plt.bar(survival_ticket.index, survival_ticket.values)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.savefig('Figure 1') # Saves image to the folder
plt.show()
#Survival Rate by Gender
plt.figure('Survival Rate 2', figsize=(7, 5))
plt.bar(survival_gender.index, survival_gender.values)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.savefig('Figure 2') # Saves image to the folder
plt.show()
#Summary Histogram
plt.figure('Age Comparison', figsize=(7, 5))
plt.hist([children['Age'].dropna(), adults['Age'].dropna()], bins = 10, color = ['blue', 'red'], edgecolor = 'white')
plt.xlabel('Age')
plt.ylabel('Passenger Count')
plt.savefig('Figure 3') # Saves image to the folder
plt.show()

#Last thoughts on this dataset
# 1. Overall survival rate shows that only a minority of passengers on board survived.
# 2. Gender had a strong impact: female passengers had a noticeably higher survival rate than male passengers.
# 3. Ticket class (Pclass) mattered a lot: 1st class passengers survived at a higher rate than 2nd and 3rd class passengers.
# 4. Age group comparison (<18 vs >=18) highlights how age distribution relates to survival outcomes.


