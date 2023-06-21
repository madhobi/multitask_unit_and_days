from run_experiments import *
     
        
def get_user_input():
        
    year_dict = {
        'a':2016,
        'b':2017,
        'c':2018,
        'd':2019,
        'e':2020,
        'f':2021
    }
    agecat_dict = {
        'a':'adult',
        'b':'child'
    }
    operation_dict = {
        'a':'train',
        'b':'test',
        'c':'both'
    }
    choice = input('\nEnter your choice: \na. Train \nb. Test \nc. Both (Train & Test)\n')
    if(choice not in operation_dict):
        print('Invalid input! Please select a,b or c')
        exit(1)    	
    else:
        print('You selected to ', operation_dict[choice])
        
    input_year = input('\nSelect a year: \na. 2016 \nb. 2017 \nc. 2018 \nd. 2019 \ne. 2020 \nf. 2021 \n') 
    
    if(input_year not in year_dict):
        print('Invalid input! Please select a,b,c,d,e or f')
        exit(1)
    else:
        print('You selected year: ', year_dict[input_year])
        
    trained_model_year = None
    if(choice == 'b'):
        input_trained_model_year = input('\nSelect year for the trained model: \na. 2016 \nb. 2017 \nc. 2018 \nd. 2019 \ne. 2020 \nf. 2021 \n')
        if(input_trained_model_year not in year_dict):
            print('Invalid input! Please select a,b,c,d,e or f')
            exit(1)
        else:
            trained_model_year = year_dict[input_trained_model_year]
            print('You selected trained model on year: ', trained_model_year)

    input_age_cat = input('\nSelect age category: a. Adult b. Children\n')
    if(input_age_cat not in agecat_dict):
        print('Invalid input! Please select a or b')
        exit(1)
    else:
        print('You selected age category is: ', agecat_dict[input_age_cat])
        
    op, year, age_cat = operation_dict[choice], year_dict[input_year], agecat_dict[input_age_cat]  

    return op, year, age_cat, trained_model_year
    

def main():
    
    operation, year, age_cat, trained_model_year =  get_user_input()
    data_filters = {
            'adm_year':year, 
            'age_cat':age_cat
        }
    
    if(operation == 'train'):
        train_model(filters=data_filters)
    elif(operation == 'test'):
        model_path = '../saved_model/model_mt_adult_year'+str(trained_model_year)        
        test_size=0.1        
        datahandler = get_datahandler(data_filters)
        _, test_pid = datahandler.get_pid(test_size=test_size)
        test_model(datahandler, test_pid, trained_model_year, model_path=model_path)
    elif(operation == 'both'):
        train_model(filters=data_filters, test=True, test_size=0.3)
    else:
        Print('Invalid selection, exiting from program..')
        exit(1)    
    
    
    
if __name__ == '__main__':
    
    main()
    
    
