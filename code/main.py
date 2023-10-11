from run_experiments import *
     
        
def get_user_input():
        
    year_dict = {
        'a':'precov',
        'b':'postcov'       
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
    op_choice = input('\nEnter your choice: \na. Train \nb. Test \nc. Both (Train & Test)\n')
    if(op_choice not in operation_dict):
        print('Invalid input! Please select a,b or c')
        exit(1)    	
    else:
        print('You selected to ', operation_dict[op_choice])
        
    input_year = input('\nSelect timeline: \na. Pre-COVID \nb. Post-COVID\n') 
    
    if(input_year not in year_dict):
        print('Invalid input! Please select a or b')
        exit(1)
    else:
        print('You selected timeline: ', year_dict[input_year], ' for ', operation_dict[op_choice])
        
    trained_model_year = None
    if(op_choice == 'b'):       
        trained_model_year = year_dict[input_year]
        print('Loading trained model fora ', trained_model_year)

    input_age_cat = input('\nSelect age category: a. Adult b. Children\n')
    if(input_age_cat not in agecat_dict):
        print('Invalid input! Please select a or b')
        exit(1)
    else:
        print('You selected age category ', agecat_dict[input_age_cat])
        
    op, year, age_cat = operation_dict[op_choice], year_dict[input_year], agecat_dict[input_age_cat]  

    return op, year, age_cat, trained_model_year
    

def main():
    
    operation, year, age_cat, trained_model_year =  get_user_input()
    # year, age_cat, operation = 'precov', 'adult', 'train'
    data_filters = {
            'adm_year':year, 
            'age_cat':age_cat
        }
    
    if(operation == 'train'):
        train_model(filters=data_filters)
    elif(operation == 'test'):
        model_path = '../saved_model/model_mt_adult_year_'+str(trained_model_year)        
        test_size=1 # take the full dataset for test      
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
    
    
