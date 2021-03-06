# Jacks-Car-Rental
Solving Reinforcement Learning - Policy Iteration Problem with Python

# Speed
Takes 13.4 sec for MAX_CAR = 20 to converge on Ryzen 2700x

# Requirements
* Python >= 3.6
* matplotlib
* scipy
* numpy

# Problem Details
* Order: move(action) -> rent -> return
* (s1, s2) = Number of cars at the location 1 and 2<br><br>

* mv = Number of cars moved from 1 to 2
* -MAX_MOVE <= mv <= MAX_MOVE
* mv < 0 means cars are moved from 2 to 1
* mv is allowed only if 0 <= s1 - mv <= MAX_CAR and 0 <= s2 + mv <= MAX_CAR<br><br>

* (rented_1, rented_2) = Number of cars rented
* rented_1 = min(rent_requests_1, s1 - mv)
* rented_2 = min(rent_requests_2, s2 + mv)<br><br>

* (returned_1, returned_2) = Number of cars returned
* returned_1 = min(return_requests_1, MAX_CAR - (s1 - mv - rented_1))
* returned_2 = min(return_requests_2, MAX_CAR - (s2 + mv - rented_2))

# Implementation Details
![](https://github.com/aask1357/Jacks-Car-Rental/blob/main/Implementation.jpg)

# Results
![](https://github.com/aask1357/Jacks-Car-Rental/blob/main/policy_5.png) ![](https://github.com/aask1357/Jacks-Car-Rental/blob/main/value_5.png)
