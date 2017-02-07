import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


def country_cont_map():

    '''returns a mapping between continents and countries'''
    countries = pd.read_csv('./Fraud/iso3166.csv')
    continents = pd.read_csv('./Fraud/country_continent.csv')

    country_cont = pd.merge(countries, continents, how='left', on='Code')
    country_cont.fillna('NA', inplace=True)
    return country_cont


class data(object):
    '''prepares data for eda, plotting, and model use'''
    def make(self):
        self.country_ip = pd.read_csv('./Fraud/IpAddress_to_Country.csv')[:1000]
        self.transact = pd.read_csv('./Fraud/Fraud_Data.csv')[:1000]
        self.transact['ip_address'] = self.transact['ip_address'].apply(lambda x: int(x))
        self.add_countries()
        self.transact.to_csv('data_with_countries.csv')

    def load(self):
        self.transact = pd.read_csv('./Fraud/data_with_countries.csv')
        return self.transact

    def save(self):
        self.transact.to_csv('data_with_countries.csv')

    def add_countries(self):
        '''create column for country by mapping to the ip address table'''
        self.transact['country'] = self.transact['ip_address'].apply(lambda x: self.get_country_from_IP(x))

    def get_country_from_IP(self, ip=None):
        '''function to match ip with country'''
        match = (self.country_ip['lower_bound_ip_address'] <= ip) & (self.country_ip['upper_bound_ip_address'] >= ip)

        if match.any():
            return self.country_ip['country'][match].to_string(index=False)
        else:
            return 'unknown'

    def clean(self):
        '''calculate account age, bin ranges for account age, customer age, purchase values, merge continent data'''
        self.transact['purchase_time'] = pd.to_datetime(self.transact['purchase_time'])

        self.transact['signup_time'] = pd.to_datetime(self.transact['signup_time'])

        self.transact['account_age'] = (self.transact['purchase_time'] - self.transact['signup_time']).apply(
            lambda x: x.days)

        self.transact['account_age_cat'] = pd.qcut(self.transact['account_age'], 4)

        self.transact['age_range'] = pd.qcut(self.transact['age'], 5)
        self.transact['purchase_range'] = pd.qcut(self.transact['purchase_value'], 4)

        self.transact = pd.merge(self.transact, country_cont_map(), how='left', left_on='country', right_on='Country')
        self.transact.fillna('Unknown', inplace=True)
        self.transact.drop('Country', inplace=True, axis=1)
        return self.transact

    def model_inputs(self):
        '''ouput dummied features for model'''
        self.inputs = self.transact[['source', 'sex', 'Continent', 'account_age_cat', 'age_range']]

        self.inputs = pd.get_dummies(self.inputs)
        return self.inputs

    def model_outputs(self):
        '''return classifications to train and test model'''
        return self.transact['class']

    def columns(self):
        return list(self.transact.columns)

    def pivot_table(self, index=None, value=None, columns=None, function=None):

        return pd.pivot_table(index=index, values=value, columns=columns, aggfunc=function)


def plot_country_fraud():
    '''plot top ten countries contribution to overall fraud'''
    d = data()
    d.load()

    d.clean()

    total_frauds = d.transact['class'].sum()
    countries = pd.pivot_table(d.transact, index='country', values='user_id', columns='class', aggfunc='count',
                               fill_value=0, margins=True, margins_name='Total')
    countries.columns = ['Not Fraud', 'Fraud', 'Total']
    countries = countries[countries.index != 'Total']

    countries['fraud_rate'] = countries['Fraud'] / countries['Total']

    countries['fraud_percent_total'] = countries['Fraud'] / total_frauds
    countries.sort(columns='fraud_percent_total', ascending=False, inplace=True)
    countries['cum_percent'] = countries['fraud_percent_total'].cumsum()

    top = 10
    # print countries['cum_percent'][:top].values

    f = plt.figure(2, figsize=(25, 15))
    f.suptitle('Fig. 2: Country contribution to fraud', fontsize=26)

    ax = f.add_subplot(111)
    ax.set_ylabel('percent of total fraud ', fontsize=15)
    # ax.plot(np.linspace(start=0,stop=19,num=20),countries['cum_percent'][:top].values)
    ax.fill_between(np.arange(0,top+1,1), countries['cum_percent'][:top+1].values, label='cumulative percent of fraud',
                    color=[.7, .7, .7])

    ax.bar(np.arange(-.4,top-.4,1), countries['fraud_percent_total'][:top].values, label='percent of total fraud', color=[.5, 0, .2])
    plt.xticks(range(top), countries.index[:top], rotation=45, fontsize=16)
    plt.legend(loc='upper left', fontsize=20)
    ax2 = ax.twinx()
    ax2.set_ylabel('country fraud rate', fontsize=15)

    ax2.plot(range(top), countries['fraud_rate'][:top], label='country fraud rate', color='red', linestyle='dashed')

    plt.legend(loc='upper center', fontsize=20)
    plt.grid(b=None, which='both')
    # plt.set_cmap('hot')
    plt.show()


def fraud_histgram():
    '''plot purchase value for records with fraud class'''
    d = data()
    d.load()

    f = plt.figure(3, figsize=(25, 15))
    f.suptitle('Figure 3: Fraud purchase value cumulative distribution', fontsize=26)
    ax = f.add_subplot(111)
    plt.hist(d.transact['purchase_value'][d.transact['class'] == 1], alpha=.7, cumulative=True, normed=True)
    plt.xlabel('Purchase Value ($)', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()


def heat_maps():
    '''make pivot tables across several variables to present in seaborn heat map'''
    d = data()
    d.load()

    d.clean()

    f = plt.figure(1, figsize=(25, 15))
    f.suptitle('Figure 1: Transaction fraud rates(descending order) vs Category', fontsize=26)
    ct = 1

    # print d.transact.columns
    index = [u'source', u'sex', 'country', u'Continent', u'account_age_cat', u'age_range', u'purchase_range']

    for i in index:
        df = pd.pivot_table(d.transact, index=i, values='user_id', columns='class',
                            aggfunc='count', fill_value=0, margins=True)
        df['perc'] = df[1] / df['All']
        df.sort(columns='perc', ascending=False, inplace=True)

        df = df[df[1] > 10]

        # print df[:30]
        ax = f.add_subplot(3, 3, ct)
        plt.title('{} vs fraud percent '.format(i), fontsize=22)

        sb.set(font_scale=1.4)

        sb.heatmap(df[['Country','perc']])
        #sb.heatmap(df[['perc']][:20], xticklabels=False)
        plt.yticks(fontsize=15, rotation=0)

        plt.xlabel('% Fraud', fontsize=15)
        plt.ylabel('', fontsize=20)

        ct += 1
    plt.subplots_adjust(wspace=.4)
    plt.show()


def classifier():
    '''actual model enclosed, produces probabilities of fraud, and purchase values for the economics'''
    d = data()
    d.load()
    d.clean()
    x = d.model_inputs()
    y = d.model_outputs()

    indices = x.index

    indices_train, indices_test = train_test_split(indices, test_size=.30)

    xtrain = x.iloc[indices_train]
    ytrain = y.iloc[indices_train]
    xtest = x.iloc[indices_test]
    ytest = y.iloc[indices_test]
    purchase_train = d.transact['purchase_value'].iloc[indices_train]
    purchase_test = d.transact['purchase_value'].iloc[indices_test]

    model = RandomForestClassifier(class_weight='balanced', n_estimators=100, oob_score=False)
    # model=LogisticRegression(class_weight='balanced')#class_weight='balanced')
    # model=SGDClassifier(class_weight='balanced')
    model.fit(xtrain, ytrain)

    indexes = np.argsort(model.feature_importances_)[::-1]
    '''feature importances'''
    print [x.columns[i] for i in indexes]

    probas_predict = model.predict_proba(xtest)[:, 1]
    ypredict = model.predict(xtest)

    '''scoring metrics'''
    print 'confusion matrix'
    print confusion_matrix(ytest, ypredict)

    print 'f1-score'
    print f1_score(ytest, ypredict)

    return ytest.values, probas_predict, purchase_test



def classifier_costs():
    '''calculates average fraud costs associated with different thresholds of the model'''
    ytest, probas, purchase_test = classifier()

    '''loss occurs for false negative scenario'''
    loss_matrix = np.array([[0, 0],
                            [1, 0]])


    admin_cost = 5
    '''admin costs incurred for any positive prediction'''
    admin_cost_matrix = np.array([[0, admin_cost],
                                  [0, admin_cost]])

    thresholds = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    losses = []
    admin_costs = []
    total_costs = []

    for threshold in thresholds:
        ypredict = (probas > threshold).astype(int)

        confusion = np.stack([ytest, ypredict], axis=0)

        print confusion

        loss_calc = loss_matrix[confusion[0, :], confusion[1, :]] * purchase_test
        admin_calc = admin_cost_matrix[confusion[0, :], confusion[1, :]]


        avg_loss = np.mean(loss_calc)
        avg_admin_cost = np.mean(admin_calc)
        total_loss = avg_loss + avg_admin_cost

        losses.append(avg_loss)
        admin_costs.append(avg_admin_cost)
        total_costs.append(total_loss)

    print thresholds
    print losses
    print admin_costs
    print total_costs


    f = plt.figure(4, figsize=(25, 15))
    f.suptitle(
        'Fig. 4: Fraud Cost per Customer vs Classifier Threshold (${:.2f} admin cost / incident)'.format(admin_cost),
        fontsize=26)
    ax = f.add_subplot(111)
    plt.stackplot(thresholds, [losses, admin_costs], baseline='zero', labels=['direct fraud costs', 'admin costs'],
                  alpha=.6)
    plt.xlabel('Classifier Threshold    ', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Cost per Customer $', fontsize=16)
    plt.legend(fontsize=16)

    xpoint = thresholds[np.argmin(total_costs)]
    ypoint = min(total_costs)

    #
    plt.annotate('', xy=(xpoint, ypoint), xytext=(xpoint - .1, ypoint + .6),
                 arrowprops=dict(facecolor='black', shrink=0.05, connectionstyle='arc3'), fontsize=16
                 )

    bbox_props = dict(boxstyle="round4,pad=0.4", fc="yellow", ec="b", lw=3, alpha=.6)
    t = ax.text(xpoint - .1, ypoint + .7, "Minimum Cost ${:.2f}".format(ypoint), ha="center", va="center", rotation=0,
                size=18, fontsize=18,
                bbox=bbox_props)

    plt.show()


def plot_countries():
    d = data()
    d.load()
    d.clean()

    print pd.pivot_table(d.transact, index=['Continent', 'country'], columns='class',
                         aggfunc='count', values='user_id', margins=True, margins_name='Total', fill_value=0
                         )

def barplots():
    '''make pivot tables across several variables to present in seaborn heat map'''
    d = data()
    d.load()

    d.clean()

    f = plt.figure(1, figsize=(25, 15))
    f.suptitle('Figure 1: Transaction fraud vs Category', fontsize=26)
    ct = 1

    # print d.transact.columns
    index = [u'source', u'sex', u'Continent', u'account_age_cat', u'age_range', u'purchase_range']

    for i in index:
        ax = f.add_subplot(2, 3, ct)
        plt.title('{} vs fraud percent '.format(i), fontsize=22)

        sb.set(font_scale=1.4)

        sb.barplot(x=i,y='class',data=d.transact, order =None)
        #sb.heatmap(df[['perc']][:20], xticklabels=False)
        plt.yticks(fontsize=15, rotation=0)

        plt.ylabel('fraud rate', fontsize=20)

        ct += 1
    plt.subplots_adjust(hspace=.4)
    plt.show()


if __name__ == '__main__':
    plot_country_fraud()
    exit()
    barplots()

    heat_maps()
    fraud_histgram()
    classifier_costs()
    classifier()
    plot_countries()
