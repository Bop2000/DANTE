# Pipeline

**To run the self-driving laboratory for compositionally complex alloy design using DANTE, see `notebooks/DANTE_VL_CCAs_Design.ipynb` for detailed instruction**

**Note:** If you can not open it in Github, please download it.

####################################################################################

To have a quick start, please run the following line in terminal:

```shell
bash run.sh
```
**Note:** The environment name should be replaced by yours in `run.sh`. You can change the number of iteration by replace `1` in `python3 run.py --iter 1`

We also provide a summary and key differences from other tasks.
1) This tasks aim at optimizing the electric properties (aha and ahc)  multi-component alloy design by desiging the composition.
2) Whats new comparing to other tasks -> helper functions to generate valid composition (external constraints); multi-objective.
3) Note -> no validation source included

# Helper function

## 1. composition limitations for all elements
```python
def component_generate():
    ele_range = []
    ele_range4 = np.arange(Co_low, Co_up, step_size) #Co
    ele_range7 = np.arange(Ni_low, Ni_up, step_size) #Ni
    ele_range18 = np.arange(Fe_low, Fe_up, step_size) #Fe
    #set other elements' component
    index_other = np.setdiff1d(np.arange(n_dim), [4,7,18])
    for i in range(n_dim-3):
        exec(f"ele_range{index_other[i]} = np.arange(other_low, other_up, step_size)")

    for i in range(n_dim):
        exec(f"ele_range.append(ele_range{i})")
    ele_range = np.array(ele_range)
    return ele_range
```
## 2. check the ratio of Fe/(Co+Ni)
```python
def FCN_ratiocheck(x_input):
    x_output = np.array(x_input)
    Fe_num = x_input[18]
    Co_num = x_input[4]
    Ni_num = x_input[7]
    sum_FCN = Fe_num + Co_num + Ni_num
    ratio = 1.5
    Fe = np.round(ratio * sum_FCN / (ratio + 1),1)
    Co = np.round((sum_FCN - Fe) * (Co_num/(Co_num + Ni_num)),1)
    Ni = np.round((sum_FCN - Fe) * (Ni_num/(Co_num + Ni_num)),1)
    x_output[18] = Fe
    x_output[7] = Ni
    x_output[4] = Co
    return x_output
```

## 3. check if Fe Co Ni has proper component, if not, adjust to a proper range
```python
def sumFCN_adjust(x_input):
    if round(x_input[4],1) == 0 and round(x_input[7],1) == 0 and round(x_input[18],1) == 0:
        x_input[4] = x_input[7] = x_input[18] = 1
        x_input[np.argmax(x_input)] -=3

    x_output = np.array(x_input)
    index_FCN = [4,7,18]
    ele_exist = np.where(x_output > 0)
    index_other = np.setdiff1d(ele_exist[0], index_FCN)
    sum_FCN = x_input[4] + x_input[7] + x_input[18]
    sum_other = sum(x_input[index_other])

    if sum_FCN > FCN_high:
        percentage = FCN_high
    elif sum_FCN <FCN_low:
        percentage = FCN_low

    for i in range(3):
        x_output[index_FCN[i]] = np.round(x_output[index_FCN[i]] * percentage / sum_FCN,1)

    for i in range(len(index_other)):
        x_output[index_other[i]] = np.round(x_output[index_other[i]] * (100 - percentage) / sum_other,1)
    return x_output
```

## 4. mode0: change element
```python
def mode0(x_input,x_output,ele_exist,index_except):
    num_to_change = np.random.randint(1, 3)
    index_FCN = np.array([4,7,18])
    index_other = np.setdiff1d(ele_exist[0], index_FCN)
    index_except = np.setdiff1d(np.arange(n_dim), ele_exist[0])
    index_other_new = np.array(index_other)
    np.random.shuffle(index_other_new)
    other_to_be_added = np.array(index_except)
    np.random.shuffle(other_to_be_added)
    for i in range(num_to_change):
        x_output[other_to_be_added[i]] = x_input[index_other_new[i]]
        x_output[index_other_new[i]] = 0
    return x_output
```
## 5. mode1: change component
```python
def mode1(x_input,x_output,ele_exist,index_except):
    num_to_change = np.random.randint(1, 3)
    ele_exist_new = np.array(ele_exist[0])
    np.random.shuffle(ele_exist_new)
    for i in range(num_to_change):
        flip = np.random.randint(0,4)
        if flip == 0:
            temp = np.random.randint(1,11)
            x_output[ele_exist_new[i]] += step_size*temp
        elif flip == 1:
            temp = np.random.randint(1,11)
            x_output[ele_exist_new[i]] -= step_size*temp
            if x_output[ele_exist_new[i]] <= 0:
                x_output[ele_exist_new[i]] += 2*temp*step_size
        else:
            x_output[ele_exist_new[i]] = ele_range[ele_exist_new[i]][np.random.randint(0, len(ele_range[ele_exist_new[i]]))]
    return x_output
```

## 6. prpose new CCAs
```python
def create_new(x_input, num_ele):
    x_output = np.array(x_input).reshape(-1)
    ele_exist = np.where(x_input > 0)#index of existed elements
    index_except = np.setdiff1d(np.arange(n_dim), ele_exist[0])

    #3 modes : change element/change component/change both
    mode = np.random.randint(0,3)

    if mode == 0:
        x_output = mode0(x_input,x_output,ele_exist,index_except)

    elif mode == 1:
        x_output = mode1(x_input,x_output,ele_exist,index_except)

    elif mode == 2:
        x_output = mode0(x_input,x_output,ele_exist,index_except)
        x_output = mode1(x_input,x_output,ele_exist,index_except)

    new_exist = np.where(x_output > 0)
    num_exist_new = len(new_exist[0])
    sum_all = sum(x_output)
    for i in range(num_exist_new):
        x_output[new_exist[0][i]] = np.round(x_output[new_exist[0][i]]*100/sum_all/step_size)*step_size

    #make sure the sum is 100
    deviation = 100 - sum(x_output)
    if deviation != 0:
        ind_max = np.argmax(x_output)
        x_output[ind_max] += deviation

    #check if the new input is proper
    sum_FCN = x_output[4] + x_output[7] + x_output[18]
    if sum_FCN > FCN_high or sum_FCN < FCN_low:
        x_output = sumFCN_adjust(x_output)

    FCN_ratio = x_output[18] - 1.5*(x_output[4] + x_output[7])
    if FCN_ratio > 0:
        x_output = FCN_ratiocheck(x_output)

    x_output = ele_num_check(x_output, num_ele)
    return x_output
```

## 7. check the number of elements
```python
def ele_num_check(x_input, ele_num):
    x_output = np.array(x_input)
    x_output[np.where(x_input < 0)[0]] = 0 ########make sure every element no less than 0
    index = np.setdiff1d(np.arange(n_dim), np.array([4,7,18]))
    ele_true = np.where(x_input > 0)
    free_ele_exist = np.intersect1d(index, ele_true[0]) #index of elements existed except Fe, Co, Ni
    free_ele_except = np.setdiff1d(index, ele_true[0]) #index of elements excepted

    if len(ele_true[0]) > ele_num:
        dev = len(ele_true[0]) - ele_num #the deviation between ele_num and true ele_num
        index_deleted = np.array(free_ele_exist)
        np.random.shuffle(index_deleted)
        for i in range(dev):
            x_output[index_deleted[i]] = 0

    elif len(ele_true[0]) < ele_num:
        dev = ele_num - len(ele_true[0])
        index_added = np.array(free_ele_except)
        np.random.shuffle(index_added)
        for i in range(dev):
            x_output[index_added[i]] = step_size

    x_output = np.round(x_output*100/(sum(x_output)*step_size))*step_size
    deviation = 100 - sum(x_output)
    if deviation != 0:
        ind_max = np.argmax(x_output)
        x_output[ind_max] += deviation
    return x_output
 ```

