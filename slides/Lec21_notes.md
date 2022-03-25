## Exercise 1

### 1. The total costs in payroll for this company

```sql
SELECT sum(salary) FROM employees;
```

### 2. The average salary within each department

```sql
SELECT avg(salary) FROM employees GROUP BY department;
```


## Exercise 2


```sqlite
sqlite> SELECT *, salary-avg AS diff FROM employees NATURAL JOIN  (SELECT dept, ROUND(AVG(salary),2) AS avg FROM employees GROUP BY dept);

name        email              salary      dept        avg         diff      
----------  -----------------  ----------  ----------  ----------  ----------
Alice       alice@company.com  52000.0     Accounting  41666.67    10333.33  
Bob         bob@company.com    40000.0     Accounting  41666.67    -1666.67  
Carol       carol@company.com  30000.0     Sales       37000.0     -7000.0   
Dave        dave@company.com   33000.0     Accounting  41666.67    -8666.67  
Eve         eve@company.com    44000.0     Sales       37000.0     7000.0    
Frank       frank@comany.com   37000.0     Sales       37000.0     0.0 
```

## Exercise 3

### Write a query that determines the total number of seats available on all of the planes that flew out of New York in 2013.

```sql
sqlite> SELECT sum(seats) FROM flights NATURAL LEFT JOIN planes;

sum(seats)
----------
614366    
Run Time: real 0.148 user 0.139176 sys 0.007804
```

--

Join and select:

```sql
sqlite> SELECT sum(seats) FROM flights LEFT JOIN planes USING (tailnum);

sum(seats)
----------
38851317  
Run Time: real 0.176 user 0.167993 sys 0.007354
```