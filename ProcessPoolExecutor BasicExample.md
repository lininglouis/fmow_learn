**ProcessPoolExecuter**.<br>
Whenever it uses a for loop or generator, we could use multiple processes to deal with the task for each generator. That is the occassion the multiple processing can be applied.


```
import concurrent.futures
import math
import timeit

def multiply_2(n):
    return n*2
 
def exampleUsing_Submit(): 
    list_num = [11,  12,    13,    14,    15,  17]
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)
    futures = [ executor.submit(multiply_2, num)     for num in list_num   ]
    results = [ fu.result() for fu in futures]
    return results
     
def exampleUsing_Map():  
    list_num = [11,  12,    13,    14,    15,  17]
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)
    results = [ res  for res in executor.map(multiply_2, list_num) ]
    return results

def exampleUsing_Nothing():  
    list_num = [11,  12,    13,    14,    15,  17]
    results = [ multiply_2(res)  for res in list_num ]
    return results
    
if __name__ == '__main__':
   
  exampleUsing_Submit()
  exampleUsing_Map()
  exampleUsing_Nothing()
  
  
  timeit.timeit(exampleUsing_Submit, number=10)
  timeit.timeit(exampleUsing_Map, number=10)
  timeit.timeit(exampleUsing_Nothing, number=10)
```


Since it is a simple task, the function without using any multiple processing is fast.<br>
You may try it with more complicated task, and compare the speed. <br>
