---
permalink: /search/Hill-climbing/
header:
  image: "/images/digital-transition2.jpg"
---
<h2 id="top"></h2>


### Random-Mutation Hill-Climbing (RMHC)

1. Choose a string at random. Call this string best-evaluated.
2. If the optimum has been found, stop and return it. If max evaluations has been
equaled or exceeded, stop and return the current value of best-evaluated. Otherwise
go to step 3.
3. Choose a locus at random to mutate. If the mutation leads to an equal or higher
fitness, then set best-evaluated to the resulting string, and go to step 2.


```python
import urllib2  # the lib that handles the url stuff
import numpy as np
import pandas as pd
#from random import randint
import random

input_data = []
url = "http://www.cs.stir.ac.uk/~goc/source/easy20.txt"
data = urllib2.urlopen(url) # it's a file like object and works just like a file
for line in data: # files are iterable
    input_data.append(map(int,line.split()))

instance_number = input_data.pop(0)[0]
max_capacity = input_data.pop()[0]
df = pd.DataFrame(input_data, columns=['no.', 'weight', 'value'])
df["weight*value"] = df["weight"]*df["value"]
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>no.</th>
      <th>weight</th>
      <th>value</th>
      <th>weight*value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>91</td>
      <td>29</td>
      <td>2639</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>60</td>
      <td>65</td>
      <td>3900</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>61</td>
      <td>71</td>
      <td>4331</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>9</td>
      <td>60</td>
      <td>540</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>79</td>
      <td>45</td>
      <td>3555</td>
    </tr>
  </tbody>
</table>
</div>




```python
#function to generate a set of binary bits to represent the items that get selected.
def binrep(n,r):
    return "{0:0{1}b}".format(n, r)
#random.randint(1, 2**instance_number)
items_selected = np.array(map(int,binrep(random.randint(1, 2**instance_number), instance_number)))
weights = np.array(df["weight"])   
values = np.array(df["value"])
```


```python
cost = 0
best_value = 0
max_eval = 1000
print "max weight = ", max_capacity, "\n"

while max_eval >0:
    total_value = np.dot(values,items_selected)
    total_weight = np.dot(weights,items_selected)
    if total_weight <= max_capacity:
        if total_value > best_value:
            best_value = total_value
    idx = random.randint(0,19)
    items_selected[idx] = int(not items_selected[idx])
    max_eval -=1
    print "current total value = ", total_value
    print "current best value = ",best_value
    print "total weight = ",total_weight, "\n"
```

    max weight =  524

    current total value =  673
    current best value =  0
    total weight =  563

    current total value =  762
    current best value =  0
    total weight =  594

    current total value =  844
    current best value =  0
    total weight =  694

    current total value =  754
    current best value =  0
    total weight =  639

    current total value =  724
    current best value =  0
    total weight =  558

    current total value =  814
    current best value =  0
    total weight =  613

    current total value =  754
    current best value =  0
    total weight =  604

    current total value =  665
    current best value =  0
    total weight =  573

    current total value =  636
    current best value =  636
    total weight =  482

    current total value =  638
    current best value =  638
    total weight =  510

    current total value =  632
    current best value =  638
    total weight =  502

    current total value =  692
    current best value =  692
    total weight =  511

    current total value =  737
    current best value =  692
    total weight =  590

    current total value =  640
    current best value =  692
    total weight =  533

    current total value =  569
    current best value =  692
    total weight =  487

    current total value =  575
    current best value =  692
    total weight =  495

    current total value =  640
    current best value =  692
    total weight =  555

    current total value =  670
    current best value =  692
    total weight =  636

    current total value =  668
    current best value =  692
    total weight =  608

    current total value =  693
    current best value =  692
    total weight =  651

    current total value =  764
    current best value =  692
    total weight =  697

    current total value =  793
    current best value =  692
    total weight =  788

    current total value =  748
    current best value =  692
    total weight =  709

    current total value =  742
    current best value =  692
    total weight =  701

    current total value =  761
    current best value =  692
    total weight =  728

    current total value =  712
    current best value =  692
    total weight =  696

    current total value =  690
    current best value =  692
    total weight =  677

    current total value =  633
    current best value =  692
    total weight =  657

    current total value =  568
    current best value =  692
    total weight =  597

    current total value =  477
    current best value =  692
    total weight =  513

    current total value =  452
    current best value =  692
    total weight =  470

    current total value =  549
    current best value =  692
    total weight =  527

    current total value =  574
    current best value =  692
    total weight =  570

    current total value =  580
    current best value =  692
    total weight =  578

    current total value =  483
    current best value =  692
    total weight =  521

    current total value =  485
    current best value =  692
    total weight =  549

    current total value =  576
    current best value =  692
    total weight =  633

    current total value =  505
    current best value =  692
    total weight =  587

    current total value =  550
    current best value =  692
    total weight =  666

    current total value =  544
    current best value =  692
    total weight =  658

    current total value =  454
    current best value =  692
    total weight =  603

    current total value =  363
    current best value =  692
    total weight =  519

    current total value =  460
    current best value =  692
    total weight =  576

    current total value =  466
    current best value =  692
    total weight =  584

    current total value =  556
    current best value =  692
    total weight =  639

    current total value =  485
    current best value =  692
    total weight =  578

    current total value =  479
    current best value =  692
    total weight =  570

    current total value =  454
    current best value =  692
    total weight =  527

    current total value =  394
    current best value =  692
    total weight =  518

    current total value =  400
    current best value =  692
    total weight =  526

    current total value =  370
    current best value =  692
    total weight =  445

    current total value =  392
    current best value =  692
    total weight =  464

    current total value =  481
    current best value =  692
    total weight =  495

    current total value =  436
    current best value =  692
    total weight =  416

    current total value =  417
    current best value =  692
    total weight =  389

    current total value =  320
    current best value =  692
    total weight =  332

    current total value =  350
    current best value =  692
    total weight =  413

    current total value =  447
    current best value =  692
    total weight =  470

    current total value =  425
    current best value =  692
    total weight =  451

    current total value =  447
    current best value =  692
    total weight =  470

    current total value =  357
    current best value =  692
    total weight =  415

    current total value =  414
    current best value =  692
    total weight =  435

    current total value =  474
    current best value =  692
    total weight =  507

    current total value =  377
    current best value =  692
    total weight =  450

    current total value =  355
    current best value =  692
    total weight =  431

    current total value =  446
    current best value =  692
    total weight =  515

    current total value =  468
    current best value =  692
    total weight =  534

    current total value =  493
    current best value =  692
    total weight =  577

    current total value =  468
    current best value =  692
    total weight =  534

    current total value =  539
    current best value =  692
    total weight =  580

    current total value =  629
    current best value =  692
    total weight =  635

    current total value =  572
    current best value =  692
    total weight =  615

    current total value =  669
    current best value =  692
    total weight =  672

    current total value =  579
    current best value =  692
    total weight =  617

    current total value =  639
    current best value =  692
    total weight =  626

    current total value =  637
    current best value =  692
    total weight =  598

    current total value =  686
    current best value =  692
    total weight =  630

    current total value =  637
    current best value =  692
    total weight =  598

    current total value =  727
    current best value =  692
    total weight =  653

    current total value =  645
    current best value =  692
    total weight =  553

    current total value =  716
    current best value =  692
    total weight =  614

    current total value =  645
    current best value =  692
    total weight =  553

    current total value =  555
    current best value =  692
    total weight =  498

    current total value =  604
    current best value =  692
    total weight =  530

    current total value =  544
    current best value =  692
    total weight =  521

    current total value =  453
    current best value =  692
    total weight =  437

    current total value =  498
    current best value =  692
    total weight =  516

    current total value =  563
    current best value =  692
    total weight =  576

    current total value =  474
    current best value =  692
    total weight =  545

    current total value =  556
    current best value =  692
    total weight =  645

    current total value =  581
    current best value =  692
    total weight =  688

    current total value =  521
    current best value =  692
    total weight =  616

    current total value =  499
    current best value =  692
    total weight =  597

    current total value =  556
    current best value =  692
    total weight =  617

    current total value =  485
    current best value =  692
    total weight =  571

    current total value =  574
    current best value =  692
    total weight =  602

    current total value =  525
    current best value =  692
    total weight =  570

    current total value =  544
    current best value =  692
    total weight =  597

    current total value =  538
    current best value =  692
    total weight =  589

    current total value =  544
    current best value =  692
    total weight =  597

    current total value =  635
    current best value =  692
    total weight =  681

    current total value =  695
    current best value =  692
    total weight =  753

    current total value =  598
    current best value =  692
    total weight =  696

    current total value =  669
    current best value =  692
    total weight =  757

    current total value =  766
    current best value =  692
    total weight =  814

    current total value =  701
    current best value =  692
    total weight =  754

    current total value =  610
    current best value =  692
    total weight =  670

    current total value =  521
    current best value =  692
    total weight =  639

    current total value =  581
    current best value =  692
    total weight =  648

    current total value =  583
    current best value =  692
    total weight =  676

    current total value =  581
    current best value =  692
    total weight =  648

    current total value =  484
    current best value =  692
    total weight =  591

    current total value =  427
    current best value =  692
    total weight =  571

    current total value =  402
    current best value =  692
    total weight =  528

    current total value =  493
    current best value =  692
    total weight =  612

    current total value =  558
    current best value =  692
    total weight =  672

    current total value =  539
    current best value =  692
    total weight =  645

    current total value =  479
    current best value =  692
    total weight =  573

    current total value =  498
    current best value =  692
    total weight =  600

    current total value =  558
    current best value =  692
    total weight =  672

    current total value =  583
    current best value =  692
    total weight =  715

    current total value =  632
    current best value =  692
    total weight =  747

    current total value =  561
    current best value =  692
    total weight =  686

    current total value =  479
    current best value =  692
    total weight =  586

    current total value =  576
    current best value =  692
    total weight =  643

    current total value =  598
    current best value =  692
    total weight =  662

    current total value =  568
    current best value =  692
    total weight =  581

    current total value =  471
    current best value =  692
    total weight =  524

    current total value =  411
    current best value =  692
    total weight =  452

    current total value =  501
    current best value =  692
    total weight =  507

    current total value =  583
    current best value =  692
    total weight =  607

    current total value =  561
    current best value =  692
    total weight =  588

    current total value =  650
    current best value =  692
    total weight =  619

    current total value =  652
    current best value =  692
    total weight =  647

    current total value =  650
    current best value =  692
    total weight =  619

    current total value =  721
    current best value =  692
    total weight =  680

    current total value =  630
    current best value =  692
    total weight =  596

    current total value =  690
    current best value =  692
    total weight =  668

    current total value =  665
    current best value =  692
    total weight =  625

    current total value =  620
    current best value =  692
    total weight =  546

    current total value =  691
    current best value =  692
    total weight =  592

    current total value =  788
    current best value =  692
    total weight =  649

    current total value =  699
    current best value =  692
    total weight =  618

    current total value =  617
    current best value =  692
    total weight =  518

    current total value =  598
    current best value =  692
    total weight =  491

    current total value =  617
    current best value =  692
    total weight =  518

    current total value =  588
    current best value =  692
    total weight =  427

    current total value =  523
    current best value =  692
    total weight =  367

    current total value =  605
    current best value =  692
    total weight =  467

    current total value =  534
    current best value =  692
    total weight =  406

    current total value =  463
    current best value =  692
    total weight =  360

    current total value =  552
    current best value =  692
    total weight =  391

    current total value =  470
    current best value =  692
    total weight =  291

    current total value =  464
    current best value =  692
    total weight =  283

    current total value =  374
    current best value =  692
    total weight =  228

    current total value =  325
    current best value =  692
    total weight =  196

    current total value =  390
    current best value =  692
    total weight =  256

    current total value =  481
    current best value =  692
    total weight =  340

    current total value =  392
    current best value =  692
    total weight =  309

    current total value =  437
    current best value =  692
    total weight =  388

    current total value =  526
    current best value =  692
    total weight =  419

    current total value =  575
    current best value =  692
    total weight =  451

    current total value =  646
    current best value =  692
    total weight =  512

    current total value =  671
    current best value =  692
    total weight =  555

    current total value =  574
    current best value =  692
    total weight =  498

    current total value =  645
    current best value =  692
    total weight =  544

    current total value =  585
    current best value =  692
    total weight =  472

    current total value =  540
    current best value =  692
    total weight =  393

    current total value =  449
    current best value =  692
    total weight =  309

    current total value =  479
    current best value =  692
    total weight =  390

    current total value =  454
    current best value =  692
    total weight =  347

    current total value =  536
    current best value =  692
    total weight =  447

    current total value =  487
    current best value =  692
    total weight =  415

    current total value =  547
    current best value =  692
    total weight =  487

    current total value =  549
    current best value =  692
    total weight =  515

    current total value =  489
    current best value =  692
    total weight =  506

    current total value =  470
    current best value =  692
    total weight =  479

    current total value =  530
    current best value =  692
    total weight =  488

    current total value =  575
    current best value =  692
    total weight =  567

    current total value =  581
    current best value =  692
    total weight =  575

    current total value =  630
    current best value =  692
    total weight =  607

    current total value =  720
    current best value =  692
    total weight =  662

    current total value =  649
    current best value =  692
    total weight =  601

    current total value =  584
    current best value =  692
    total weight =  541

    current total value =  524
    current best value =  692
    total weight =  532

    current total value =  475
    current best value =  692
    total weight =  500

    current total value =  393
    current best value =  692
    total weight =  400

    current total value =  458
    current best value =  692
    total weight =  460

    current total value =  487
    current best value =  692
    total weight =  551

    current total value =  457
    current best value =  692
    total weight =  470

    current total value =  554
    current best value =  692
    total weight =  527

    current total value =  464
    current best value =  692
    total weight =  472

    current total value =  555
    current best value =  692
    total weight =  556

    current total value =  637
    current best value =  692
    total weight =  656

    current total value =  555
    current best value =  692
    total weight =  556

    current total value =  577
    current best value =  692
    total weight =  575

    current total value =  634
    current best value =  692
    total weight =  595

    current total value =  537
    current best value =  692
    total weight =  538

    current total value =  567
    current best value =  692
    total weight =  619

    current total value =  537
    current best value =  692
    total weight =  538

    current total value =  608
    current best value =  692
    total weight =  599

    current total value =  543
    current best value =  692
    total weight =  539

    current total value =  452
    current best value =  692
    total weight =  455

    current total value =  392
    current best value =  692
    total weight =  383

    current total value =  482
    current best value =  692
    total weight =  438

    current total value =  512
    current best value =  692
    total weight =  519

    current total value =  506
    current best value =  692
    total weight =  511

    current total value =  603
    current best value =  692
    total weight =  568

    current total value =  694
    current best value =  692
    total weight =  652

    current total value =  713
    current best value =  692
    total weight =  679

    current total value =  622
    current best value =  692
    total weight =  595

    current total value =  628
    current best value =  692
    total weight =  603

    current total value =  710
    current best value =  692
    total weight =  703

    current total value =  628
    current best value =  692
    total weight =  603

    current total value =  609
    current best value =  692
    total weight =  576

    current total value =  607
    current best value =  692
    total weight =  548

    current total value =  689
    current best value =  692
    total weight =  648

    current total value =  738
    current best value =  692
    total weight =  680

    current total value =  732
    current best value =  692
    total weight =  672

    current total value =  710
    current best value =  692
    total weight =  653

    current total value =  716
    current best value =  692
    total weight =  661

    current total value =  781
    current best value =  692
    total weight =  721

    current total value =  806
    current best value =  692
    total weight =  764

    current total value =  825
    current best value =  692
    total weight =  791

    current total value =  743
    current best value =  692
    total weight =  691

    current total value =  765
    current best value =  692
    total weight =  710

    current total value =  825
    current best value =  692
    total weight =  782

    current total value =  736
    current best value =  692
    total weight =  751

    current total value =  738
    current best value =  692
    total weight =  779

    current total value =  736
    current best value =  692
    total weight =  751

    current total value =  825
    current best value =  692
    total weight =  782

    current total value =  754
    current best value =  692
    total weight =  721

    current total value =  697
    current best value =  692
    total weight =  701

    current total value =  779
    current best value =  692
    total weight =  801

    current total value =  690
    current best value =  692
    total weight =  770

    current total value =  692
    current best value =  692
    total weight =  798

    current total value =  763
    current best value =  692
    total weight =  859

    current total value =  854
    current best value =  692
    total weight =  943

    current total value =  789
    current best value =  692
    total weight =  883

    current total value =  760
    current best value =  692
    total weight =  792

    current total value =  825
    current best value =  692
    total weight =  852

    current total value =  819
    current best value =  692
    total weight =  844

    current total value =  794
    current best value =  692
    total weight =  801

    current total value =  734
    current best value =  692
    total weight =  729

    current total value =  689
    current best value =  692
    total weight =  650

    current total value =  670
    current best value =  692
    total weight =  623

    current total value =  599
    current best value =  692
    total weight =  577

    current total value =  688
    current best value =  692
    total weight =  608

    current total value =  686
    current best value =  692
    total weight =  580

    current total value =  637
    current best value =  692
    total weight =  548

    current total value =  540
    current best value =  692
    total weight =  491

    current total value =  611
    current best value =  692
    total weight =  537

    current total value =  589
    current best value =  692
    total weight =  518

    current total value =  591
    current best value =  692
    total weight =  546

    current total value =  613
    current best value =  692
    total weight =  565

    current total value =  673
    current best value =  692
    total weight =  637

    current total value =  698
    current best value =  692
    total weight =  680

    current total value =  758
    current best value =  692
    total weight =  689

    current total value =  736
    current best value =  692
    total weight =  670

    current total value =  676
    current best value =  692
    total weight =  661

    current total value =  674
    current best value =  692
    total weight =  633

    current total value =  731
    current best value =  692
    total weight =  653

    current total value =  760
    current best value =  692
    total weight =  744

    current total value =  670
    current best value =  692
    total weight =  689

    current total value =  645
    current best value =  692
    total weight =  646

    current total value =  647
    current best value =  692
    total weight =  674

    current total value =  672
    current best value =  692
    total weight =  717

    current total value =  583
    current best value =  692
    total weight =  686

    current total value =  526
    current best value =  692
    total weight =  666

    current total value =  524
    current best value =  692
    total weight =  638

    current total value =  453
    current best value =  692
    total weight =  592

    current total value =  524
    current best value =  692
    total weight =  638

    current total value =  613
    current best value =  692
    total weight =  669

    current total value =  524
    current best value =  692
    total weight =  638

    current total value =  453
    current best value =  692
    total weight =  592

    current total value =  502
    current best value =  692
    total weight =  624

    current total value =  504
    current best value =  692
    total weight =  652

    current total value =  510
    current best value =  692
    total weight =  660

    current total value =  480
    current best value =  692
    total weight =  579

    current total value =  499
    current best value =  692
    total weight =  606

    current total value =  474
    current best value =  692
    total weight =  563

    current total value =  563
    current best value =  692
    total weight =  594

    current total value =  634
    current best value =  692
    total weight =  640

    current total value =  545
    current best value =  692
    total weight =  609

    current total value =  635
    current best value =  692
    total weight =  664

    current total value =  660
    current best value =  692
    total weight =  707

    current total value =  720
    current best value =  692
    total weight =  716

    current total value =  742
    current best value =  692
    total weight =  735

    current total value =  723
    current best value =  692
    total weight =  708

    current total value =  780
    current best value =  692
    total weight =  728

    current total value =  825
    current best value =  692
    total weight =  807

    current total value =  855
    current best value =  692
    total weight =  888

    current total value =  795
    current best value =  692
    total weight =  879

    current total value =  793
    current best value =  692
    total weight =  851

    current total value =  711
    current best value =  692
    total weight =  751

    current total value =  640
    current best value =  692
    total weight =  690

    current total value =  550
    current best value =  692
    total weight =  635

    current total value =  639
    current best value =  692
    total weight =  666

    current total value =  641
    current best value =  692
    total weight =  694

    current total value =  576
    current best value =  692
    total weight =  634

    current total value =  636
    current best value =  692
    total weight =  643

    current total value =  587
    current best value =  692
    total weight =  611

    current total value =  527
    current best value =  692
    total weight =  539

    current total value =  546
    current best value =  692
    total weight =  566

    current total value =  606
    current best value =  692
    total weight =  638

    current total value =  535
    current best value =  692
    total weight =  592

    current total value =  475
    current best value =  692
    total weight =  583

    current total value =  540
    current best value =  692
    total weight =  643

    current total value =  622
    current best value =  692
    total weight =  743

    current total value =  712
    current best value =  692
    total weight =  798

    current total value =  687
    current best value =  692
    total weight =  755

    current total value =  758
    current best value =  692
    total weight =  801

    current total value =  687
    current best value =  692
    total weight =  755

    current total value =  736
    current best value =  692
    total weight =  787

    current total value =  706
    current best value =  692
    total weight =  706

    current total value =  684
    current best value =  692
    total weight =  687

    current total value =  635
    current best value =  692
    total weight =  655

    current total value =  545
    current best value =  692
    total weight =  600

    current total value =  575
    current best value =  692
    total weight =  681

    current total value =  484
    current best value =  692
    total weight =  597

    current total value =  544
    current best value =  692
    total weight =  606

    current total value =  615
    current best value =  692
    total weight =  667

    current total value =  544
    current best value =  692
    total weight =  606

    current total value =  455
    current best value =  692
    total weight =  575

    current total value =  390
    current best value =  692
    total weight =  515

    current total value =  345
    current best value =  692
    total weight =  436

    current total value =  285
    current best value =  692
    total weight =  427

    current total value =  382
    current best value =  692
    total weight =  484

    current total value =  353
    current best value =  692
    total weight =  393

    current total value =  271
    current best value =  692
    total weight =  293

    current total value =  360
    current best value =  692
    total weight =  324

    current total value =  451
    current best value =  692
    total weight =  408

    current total value =  360
    current best value =  692
    total weight =  324

    current total value =  271
    current best value =  692
    total weight =  293

    current total value =  252
    current best value =  692
    total weight =  266

    current total value =  317
    current best value =  692
    total weight =  326

    current total value =  220
    current best value =  692
    total weight =  269

    current total value =  291
    current best value =  692
    total weight =  315

    current total value =  373
    current best value =  692
    total weight =  415

    current total value =  402
    current best value =  692
    total weight =  506

    current total value =  421
    current best value =  692
    total weight =  533

    current total value =  446
    current best value =  692
    total weight =  576

    current total value =  417
    current best value =  692
    total weight =  485

    current total value =  507
    current best value =  692
    total weight =  540

    current total value =  598
    current best value =  692
    total weight =  624

    current total value =  568
    current best value =  692
    total weight =  543

    current total value =  497
    current best value =  692
    total weight =  497

    current total value =  546
    current best value =  692
    total weight =  529

    current total value =  635
    current best value =  692
    total weight =  560

    current total value =  680
    current best value =  692
    total weight =  639

    current total value =  751
    current best value =  692
    total weight =  685

    current total value =  822
    current best value =  692
    total weight =  746

    current total value =  919
    current best value =  692
    total weight =  803

    current total value =  979
    current best value =  692
    total weight =  812

    current total value =  897
    current best value =  692
    total weight =  712

    current total value =  826
    current best value =  692
    total weight =  651

    current total value =  801
    current best value =  692
    total weight =  608

    current total value =  883
    current best value =  692
    total weight =  708

    current total value =  908
    current best value =  692
    total weight =  751

    current total value =  883
    current best value =  692
    total weight =  708

    current total value =  818
    current best value =  692
    total weight =  648

    current total value =  721
    current best value =  692
    total weight =  591

    current total value =  631
    current best value =  692
    total weight =  536

    current total value =  542
    current best value =  692
    total weight =  505

    current total value =  482
    current best value =  692
    total weight =  433

    current total value =  542
    current best value =  692
    total weight =  505

    current total value =  493
    current best value =  692
    total weight =  473

    current total value =  564
    current best value =  692
    total weight =  534

    current total value =  504
    current best value =  692
    total weight =  462

    current total value =  444
    current best value =  692
    total weight =  453

    current total value =  473
    current best value =  692
    total weight =  544

    current total value =  533
    current best value =  692
    total weight =  553

    current total value =  451
    current best value =  692
    total weight =  453

    current total value =  360
    current best value =  692
    total weight =  369

    current total value =  451
    current best value =  692
    total weight =  453

    current total value =  533
    current best value =  692
    total weight =  553

    current total value =  531
    current best value =  692
    total weight =  525

    current total value =  553
    current best value =  692
    total weight =  544

    current total value =  531
    current best value =  692
    total weight =  525

    current total value =  620
    current best value =  692
    total weight =  556

    current total value =  601
    current best value =  692
    total weight =  529

    current total value =  572
    current best value =  692
    total weight =  438

    current total value =  637
    current best value =  692
    total weight =  498

    current total value =  734
    current best value =  692
    total weight =  555

    current total value =  669
    current best value =  692
    total weight =  495

    current total value =  609
    current best value =  692
    total weight =  486

    current total value =  512
    current best value =  692
    total weight =  429

    current total value =  506
    current best value =  692
    total weight =  421

    current total value =  415
    current best value =  692
    total weight =  337

    current total value =  434
    current best value =  692
    total weight =  364

    current total value =  415
    current best value =  692
    total weight =  337

    current total value =  344
    current best value =  692
    total weight =  276

    current total value =  373
    current best value =  692
    total weight =  367

    current total value =  344
    current best value =  692
    total weight =  276

    current total value =  366
    current best value =  692
    total weight =  295

    current total value =  426
    current best value =  692
    total weight =  304

    current total value =  486
    current best value =  692
    total weight =  376

    current total value =  551
    current best value =  692
    total weight =  436

    current total value =  600
    current best value =  692
    total weight =  468

    current total value =  625
    current best value =  692
    total weight =  511

    current total value =  655
    current best value =  692
    total weight =  592

    current total value =  606
    current best value =  692
    total weight =  560

    current total value =  524
    current best value =  692
    total weight =  460

    current total value =  479
    current best value =  692
    total weight =  381

    current total value =  414
    current best value =  692
    total weight =  321

    current total value =  433
    current best value =  692
    total weight =  348

    current total value =  515
    current best value =  692
    total weight =  448

    current total value =  433
    current best value =  692
    total weight =  348

    current total value =  373
    current best value =  692
    total weight =  276

    current total value =  375
    current best value =  692
    total weight =  304

    current total value =  356
    current best value =  692
    total weight =  277

    current total value =  296
    current best value =  692
    total weight =  268

    current total value =  274
    current best value =  692
    total weight =  249

    current total value =  356
    current best value =  692
    total weight =  349

    current total value =  274
    current best value =  692
    total weight =  249

    current total value =  244
    current best value =  692
    total weight =  168

    current total value =  304
    current best value =  692
    total weight =  177

    current total value =  364
    current best value =  692
    total weight =  249

    current total value =  461
    current best value =  692
    total weight =  306

    current total value =  364
    current best value =  692
    total weight =  249

    current total value =  429
    current best value =  692
    total weight =  309

    current total value =  369
    current best value =  692
    total weight =  300

    current total value =  375
    current best value =  692
    total weight =  308

    current total value =  369
    current best value =  692
    total weight =  300

    current total value =  298
    current best value =  692
    total weight =  254

    current total value =  273
    current best value =  692
    total weight =  211

    current total value =  344
    current best value =  692
    total weight =  272

    current total value =  284
    current best value =  692
    total weight =  200

    current total value =  303
    current best value =  692
    total weight =  227

    current total value =  232
    current best value =  692
    total weight =  166

    current total value =  292
    current best value =  692
    total weight =  238

    current total value =  341
    current best value =  692
    total weight =  270

    current total value =  423
    current best value =  692
    total weight =  370

    current total value =  358
    current best value =  692
    total weight =  310

    current total value =  429
    current best value =  692
    total weight =  371

    current total value =  520
    current best value =  692
    total weight =  455

    current total value =  545
    current best value =  692
    total weight =  498

    current total value =  642
    current best value =  692
    total weight =  555

    current total value =  687
    current best value =  692
    total weight =  634

    current total value =  616
    current best value =  692
    total weight =  573

    current total value =  571
    current best value =  692
    total weight =  494

    current total value =  600
    current best value =  692
    total weight =  585

    current total value =  503
    current best value =  692
    total weight =  528

    current total value =  443
    current best value =  692
    total weight =  456

    current total value =  361
    current best value =  692
    total weight =  356

    current total value =  406
    current best value =  692
    total weight =  435

    current total value =  488
    current best value =  692
    total weight =  535

    current total value =  443
    current best value =  692
    total weight =  456

    current total value =  449
    current best value =  692
    total weight =  464

    current total value =  509
    current best value =  692
    total weight =  536

    current total value =  599
    current best value =  692
    total weight =  591

    current total value =  510
    current best value =  692
    total weight =  560

    current total value =  453
    current best value =  692
    total weight =  540

    current total value =  498
    current best value =  692
    total weight =  619

    current total value =  496
    current best value =  692
    total weight =  591

    current total value =  567
    current best value =  692
    total weight =  652

    current total value =  496
    current best value =  692
    total weight =  591

    current total value =  477
    current best value =  692
    total weight =  564

    current total value =  448
    current best value =  692
    total weight =  473

    current total value =  388
    current best value =  692
    total weight =  401

    current total value =  343
    current best value =  692
    total weight =  322

    current total value =  318
    current best value =  692
    total weight =  279

    current total value =  269
    current best value =  692
    total weight =  247

    current total value =  358
    current best value =  692
    total weight =  278

    current total value =  380
    current best value =  692
    total weight =  297

    current total value =  358
    current best value =  692
    total weight =  278

    current total value =  403
    current best value =  692
    total weight =  357

    current total value =  432
    current best value =  692
    total weight =  448

    current total value =  343
    current best value =  692
    total weight =  417

    current total value =  345
    current best value =  692
    total weight =  445

    current total value =  394
    current best value =  692
    total weight =  477

    current total value =  345
    current best value =  692
    total weight =  445

    current total value =  434
    current best value =  692
    total weight =  476

    current total value =  505
    current best value =  692
    total weight =  537

    current total value =  565
    current best value =  692
    total weight =  546

    current total value =  494
    current best value =  692
    total weight =  485

    current total value =  565
    current best value =  692
    total weight =  546

    current total value =  614
    current best value =  692
    total weight =  578

    current total value =  525
    current best value =  692
    total weight =  547

    current total value =  465
    current best value =  692
    total weight =  538

    current total value =  525
    current best value =  692
    total weight =  610

    current total value =  434
    current best value =  692
    total weight =  526

    current total value =  352
    current best value =  692
    total weight =  426

    current total value =  374
    current best value =  692
    total weight =  445

    current total value =  372
    current best value =  692
    total weight =  417

    current total value =  397
    current best value =  692
    total weight =  460

    current total value =  416
    current best value =  692
    total weight =  487

    current total value =  391
    current best value =  692
    total weight =  444

    current total value =  385
    current best value =  692
    total weight =  436

    current total value =  474
    current best value =  692
    total weight =  467

    current total value =  571
    current best value =  692
    total weight =  524

    current total value =  631
    current best value =  692
    total weight =  533

    current total value =  560
    current best value =  692
    total weight =  472

    current total value =  625
    current best value =  692
    total weight =  532

    current total value =  650
    current best value =  692
    total weight =  575

    current total value =  680
    current best value =  692
    total weight =  656

    current total value =  650
    current best value =  692
    total weight =  575

    current total value =  621
    current best value =  692
    total weight =  484

    current total value =  623
    current best value =  692
    total weight =  512

    current total value =  621
    current best value =  692
    total weight =  484

    current total value =  576
    current best value =  692
    total weight =  405

    current total value =  554
    current best value =  692
    total weight =  386

    current total value =  611
    current best value =  692
    total weight =  406

    current total value =  551
    current best value =  692
    total weight =  397

    current total value =  494
    current best value =  692
    total weight =  377

    current total value =  434
    current best value =  692
    total weight =  305

    current total value =  369
    current best value =  692
    total weight =  245

    current total value =  279
    current best value =  692
    total weight =  190

    current total value =  308
    current best value =  692
    total weight =  281

    current total value =  330
    current best value =  692
    total weight =  300

    current total value =  301
    current best value =  692
    total weight =  209

    current total value =  279
    current best value =  692
    total weight =  190

    current total value =  350
    current best value =  692
    total weight =  236

    current total value =  380
    current best value =  692
    total weight =  317

    current total value =  350
    current best value =  692
    total weight =  236

    current total value =  379
    current best value =  692
    total weight =  327

    current total value =  450
    current best value =  692
    total weight =  388

    current total value =  510
    current best value =  692
    total weight =  397

    current total value =  516
    current best value =  692
    total weight =  405

    current total value =  487
    current best value =  692
    total weight =  314

    current total value =  509
    current best value =  692
    total weight =  333

    current total value =  600
    current best value =  692
    total weight =  417

    current total value =  602
    current best value =  692
    total weight =  445

    current total value =  632
    current best value =  692
    total weight =  526

    current total value =  630
    current best value =  692
    total weight =  498

    current total value =  611
    current best value =  692
    total weight =  471

    current total value =  605
    current best value =  692
    total weight =  463

    current total value =  508
    current best value =  692
    total weight =  406

    current total value =  459
    current best value =  692
    total weight =  374

    current total value =  556
    current best value =  692
    total weight =  431

    current total value =  638
    current best value =  692
    total weight =  531

    current total value =  667
    current best value =  692
    total weight =  622

    current total value =  607
    current best value =  692
    total weight =  613

    current total value =  672
    current best value =  692
    total weight =  673

    current total value =  762
    current best value =  692
    total weight =  728

    current total value =  671
    current best value =  692
    total weight =  644

    current total value =  649
    current best value =  692
    total weight =  625

    current total value =  620
    current best value =  692
    total weight =  534

    current total value =  665
    current best value =  692
    total weight =  613

    current total value =  725
    current best value =  692
    total weight =  685

    current total value =  731
    current best value =  692
    total weight =  693

    current total value =  686
    current best value =  692
    total weight =  614

    current total value =  596
    current best value =  692
    total weight =  559

    current total value =  615
    current best value =  692
    total weight =  586

    current total value =  544
    current best value =  692
    total weight =  525

    current total value =  525
    current best value =  692
    total weight =  498

    current total value =  527
    current best value =  692
    total weight =  526

    current total value =  438
    current best value =  692
    total weight =  495

    current total value =  367
    current best value =  692
    total weight =  449

    current total value =  270
    current best value =  692
    total weight =  392

    current total value =  210
    current best value =  692
    total weight =  320

    current total value =  239
    current best value =  692
    total weight =  411

    current total value =  214
    current best value =  692
    total weight =  368

    current total value =  311
    current best value =  692
    total weight =  425

    current total value =  336
    current best value =  692
    total weight =  468

    current total value =  385
    current best value =  692
    total weight =  500

    current total value =  303
    current best value =  692
    total weight =  400

    current total value =  374
    current best value =  692
    total weight =  446

    current total value =  277
    current best value =  692
    total weight =  389

    current total value =  367
    current best value =  692
    total weight =  444

    current total value =  464
    current best value =  692
    total weight =  501

    current total value =  509
    current best value =  692
    total weight =  580

    current total value =  591
    current best value =  692
    total weight =  680

    current total value =  566
    current best value =  692
    total weight =  637

    current total value =  626
    current best value =  692
    total weight =  709

    current total value =  577
    current best value =  692
    total weight =  677

    current total value =  548
    current best value =  692
    total weight =  586

    current total value =  546
    current best value =  692
    total weight =  558

    current total value =  486
    current best value =  692
    total weight =  486

    current total value =  415
    current best value =  692
    total weight =  440

    current total value =  472
    current best value =  692
    total weight =  460

    current total value =  561
    current best value =  692
    total weight =  491

    current total value =  621
    current best value =  692
    total weight =  563

    current total value =  539
    current best value =  692
    total weight =  463

    current total value =  509
    current best value =  692
    total weight =  382

    current total value =  534
    current best value =  692
    total weight =  425

    current total value =  469
    current best value =  692
    total weight =  365

    current total value =  444
    current best value =  692
    total weight =  322

    current total value =  463
    current best value =  692
    total weight =  349

    current total value =  485
    current best value =  692
    total weight =  368

    current total value =  545
    current best value =  692
    total weight =  377

    current total value =  526
    current best value =  692
    total weight =  350

    current total value =  466
    current best value =  692
    total weight =  278

    current total value =  485
    current best value =  692
    total weight =  305

    current total value =  556
    current best value =  692
    total weight =  366

    current total value =  586
    current best value =  692
    total weight =  447

    current total value =  677
    current best value =  692
    total weight =  531

    current total value =  655
    current best value =  692
    total weight =  512

    current total value =  636
    current best value =  692
    total weight =  485

    current total value =  546
    current best value =  692
    total weight =  430

    current total value =  571
    current best value =  692
    total weight =  473

    current total value =  620
    current best value =  692
    total weight =  505

    current total value =  563
    current best value =  692
    total weight =  485

    current total value =  474
    current best value =  692
    total weight =  454

    current total value =  425
    current best value =  692
    total weight =  422

    current total value =  354
    current best value =  692
    total weight =  361

    current total value =  356
    current best value =  692
    total weight =  389

    current total value =  378
    current best value =  692
    total weight =  408

    current total value =  468
    current best value =  692
    total weight =  463

    current total value =  557
    current best value =  692
    total weight =  494

    current total value =  468
    current best value =  692
    total weight =  463

    current total value =  371
    current best value =  692
    total weight =  406

    current total value =  341
    current best value =  692
    total weight =  325

    current total value =  360
    current best value =  692
    total weight =  352

    current total value =  431
    current best value =  692
    total weight =  398

    current total value =  409
    current best value =  692
    total weight =  379

    current total value =  431
    current best value =  692
    total weight =  398

    current total value =  409
    current best value =  692
    total weight =  379

    current total value =  439
    current best value =  692
    total weight =  460

    current total value =  437
    current best value =  692
    total weight =  432

    current total value =  526
    current best value =  692
    total weight =  463

    current total value =  455
    current best value =  692
    total weight =  417

    current total value =  520
    current best value =  692
    total weight =  477

    current total value =  549
    current best value =  692
    total weight =  568

    current total value =  543
    current best value =  692
    total weight =  560

    current total value =  452
    current best value =  692
    total weight =  476

    current total value =  501
    current best value =  692
    total weight =  508

    current total value =  592
    current best value =  692
    total weight =  592

    current total value =  614
    current best value =  692
    total weight =  611

    current total value =  525
    current best value =  692
    total weight =  580

    current total value =  434
    current best value =  692
    total weight =  496

    current total value =  531
    current best value =  692
    total weight =  553

    current total value =  441
    current best value =  692
    total weight =  498

    current total value =  416
    current best value =  692
    total weight =  455

    current total value =  394
    current best value =  692
    total weight =  436

    current total value =  364
    current best value =  692
    total weight =  355

    current total value =  370
    current best value =  692
    total weight =  363

    current total value =  441
    current best value =  692
    total weight =  424

    current total value =  498
    current best value =  692
    total weight =  444

    current total value =  580
    current best value =  692
    total weight =  544

    current total value =  671
    current best value =  692
    total weight =  628

    current total value =  731
    current best value =  692
    total weight =  700

    current total value =  761
    current best value =  692
    total weight =  781

    current total value =  732
    current best value =  692
    total weight =  690

    current total value =  641
    current best value =  692
    total weight =  606

    current total value =  730
    current best value =  692
    total weight =  637

    current total value =  732
    current best value =  692
    total weight =  665

    current total value =  823
    current best value =  692
    total weight =  749

    current total value =  766
    current best value =  692
    total weight =  729

    current total value =  684
    current best value =  692
    total weight =  629

    current total value =  665
    current best value =  692
    total weight =  602

    current total value =  694
    current best value =  692
    total weight =  693

    current total value =  713
    current best value =  692
    total weight =  720

    current total value =  784
    current best value =  692
    total weight =  766

    current total value =  778
    current best value =  692
    total weight =  758

    current total value =  776
    current best value =  692
    total weight =  730

    current total value =  687
    current best value =  692
    total weight =  699

    current total value =  616
    current best value =  692
    total weight =  653

    current total value =  525
    current best value =  692
    total weight =  569

    current total value =  614
    current best value =  692
    total weight =  600

    current total value =  704
    current best value =  692
    total weight =  655

    current total value =  655
    current best value =  692
    total weight =  623

    current total value =  680
    current best value =  692
    total weight =  666

    current total value =  771
    current best value =  692
    total weight =  750

    current total value =  700
    current best value =  692
    total weight =  689

    current total value =  611
    current best value =  692
    total weight =  658

    current total value =  617
    current best value =  692
    total weight =  666

    current total value =  598
    current best value =  692
    total weight =  639

    current total value =  553
    current best value =  692
    total weight =  560

    current total value =  462
    current best value =  692
    total weight =  476

    current total value =  432
    current best value =  692
    total weight =  395

    current total value =  503
    current best value =  692
    total weight =  441

    current total value =  533
    current best value =  692
    total weight =  522

    current total value =  582
    current best value =  692
    total weight =  554

    current total value =  576
    current best value =  692
    total weight =  546

    current total value =  527
    current best value =  692
    total weight =  514

    current total value =  584
    current best value =  692
    total weight =  534

    current total value =  633
    current best value =  692
    total weight =  566

    current total value =  724
    current best value =  692
    total weight =  650

    current total value =  695
    current best value =  692
    total weight =  559

    current total value =  635
    current best value =  692
    total weight =  487

    current total value =  637
    current best value =  692
    total weight =  515

    current total value =  612
    current best value =  692
    total weight =  472

    current total value =  631
    current best value =  692
    total weight =  499

    current total value =  540
    current best value =  692
    total weight =  415

    current total value =  480
    current best value =  692
    total weight =  406

    current total value =  562
    current best value =  692
    total weight =  506

    current total value =  465
    current best value =  692
    total weight =  449

    current total value =  525
    current best value =  692
    total weight =  458

    current total value =  454
    current best value =  692
    total weight =  412

    current total value =  499
    current best value =  692
    total weight =  491

    current total value =  524
    current best value =  692
    total weight =  534

    current total value =  434
    current best value =  692
    total weight =  479

    current total value =  463
    current best value =  692
    total weight =  570

    current total value =  414
    current best value =  692
    total weight =  538

    current total value =  485
    current best value =  692
    total weight =  599

    current total value =  455
    current best value =  692
    total weight =  518

    current total value =  546
    current best value =  692
    total weight =  602

    current total value =  455
    current best value =  692
    total weight =  518

    current total value =  552
    current best value =  692
    total weight =  575

    current total value =  574
    current best value =  692
    total weight =  594

    current total value =  503
    current best value =  692
    total weight =  533

    current total value =  484
    current best value =  692
    total weight =  506

    current total value =  503
    current best value =  692
    total weight =  533

    current total value =  592
    current best value =  692
    total weight =  564

    current total value =  532
    current best value =  692
    total weight =  555

    current total value =  623
    current best value =  692
    total weight =  639

    current total value =  541
    current best value =  692
    total weight =  539

    current total value =  516
    current best value =  692
    total weight =  496

    current total value =  459
    current best value =  692
    total weight =  476

    current total value =  519
    current best value =  692
    total weight =  485

    current total value =  590
    current best value =  692
    total weight =  531

    current total value =  650
    current best value =  692
    total weight =  603

    current total value =  732
    current best value =  692
    total weight =  703

    current total value =  661
    current best value =  692
    total weight =  657

    current total value =  667
    current best value =  692
    total weight =  665

    current total value =  757
    current best value =  692
    total weight =  720

    current total value =  814
    current best value =  692
    total weight =  740

    current total value =  863
    current best value =  692
    total weight =  772

    current total value =  803
    current best value =  692
    total weight =  700

    current total value =  874
    current best value =  692
    total weight =  761

    current total value =  868
    current best value =  692
    total weight =  753

    current total value =  898
    current best value =  692
    total weight =  834

    current total value =  801
    current best value =  692
    total weight =  777

    current total value =  807
    current best value =  692
    total weight =  785

    current total value =  878
    current best value =  692
    total weight =  831

    current total value =  789
    current best value =  692
    total weight =  800

    current total value =  698
    current best value =  692
    total weight =  716

    current total value =  633
    current best value =  692
    total weight =  656

    current total value =  693
    current best value =  692
    total weight =  728

    current total value =  671
    current best value =  692
    total weight =  709

    current total value =  768
    current best value =  692
    total weight =  766

    current total value =  671
    current best value =  692
    total weight =  709

    current total value =  600
    current best value =  692
    total weight =  663

    current total value =  665
    current best value =  692
    total weight =  723

    current total value =  605
    current best value =  692
    total weight =  714

    current total value =  534
    current best value =  692
    total weight =  653

    current total value =  556
    current best value =  692
    total weight =  672

    current total value =  645
    current best value =  692
    total weight =  703

    current total value =  588
    current best value =  692
    total weight =  683

    current total value =  659
    current best value =  692
    total weight =  729

    current total value =  577
    current best value =  692
    total weight =  629

    current total value =  532
    current best value =  692
    total weight =  550

    current total value =  472
    current best value =  692
    total weight =  478

    current total value =  407
    current best value =  692
    total weight =  418

    current total value =  489
    current best value =  692
    total weight =  518

    current total value =  407
    current best value =  692
    total weight =  418

    current total value =  405
    current best value =  692
    total weight =  390

    current total value =  316
    current best value =  692
    total weight =  359

    current total value =  381
    current best value =  692
    total weight =  419

    current total value =  452
    current best value =  692
    total weight =  480

    current total value =  423
    current best value =  692
    total weight =  389

    current total value =  483
    current best value =  692
    total weight =  461

    current total value =  574
    current best value =  692
    total weight =  545

    current total value =  599
    current best value =  692
    total weight =  588

    current total value =  534
    current best value =  692
    total weight =  528

    current total value =  515
    current best value =  692
    total weight =  501

    current total value =  612
    current best value =  692
    total weight =  558

    current total value =  521
    current best value =  692
    total weight =  474

    current total value =  461
    current best value =  692
    total weight =  402

    current total value =  550
    current best value =  692
    total weight =  433

    current total value =  610
    current best value =  692
    total weight =  505

    current total value =  692
    current best value =  692
    total weight =  605

    current total value =  757
    current best value =  692
    total weight =  665

    current total value =  802
    current best value =  692
    total weight =  744

    current total value =  742
    current best value =  692
    total weight =  672

    current total value =  744
    current best value =  692
    total weight =  700

    current total value =  655
    current best value =  692
    total weight =  669

    current total value =  674
    current best value =  692
    total weight =  696

    current total value =  655
    current best value =  692
    total weight =  669

    current total value =  590
    current best value =  692
    total weight =  609

    current total value =  584
    current best value =  692
    total weight =  601

    current total value =  641
    current best value =  692
    total weight =  621

    current total value =  596
    current best value =  692
    total weight =  542

    current total value =  514
    current best value =  692
    total weight =  442

    current total value =  443
    current best value =  692
    total weight =  396

    current total value =  413
    current best value =  692
    total weight =  315

    current total value =  442
    current best value =  692
    total weight =  406

    current total value =  417
    current best value =  692
    total weight =  363

    current total value =  447
    current best value =  692
    total weight =  444

    current total value =  390
    current best value =  692
    total weight =  424

    current total value =  396
    current best value =  692
    total weight =  432

    current total value =  299
    current best value =  692
    total weight =  375

    current total value =  324
    current best value =  692
    total weight =  418

    current total value =  395
    current best value =  692
    total weight =  464

    current total value =  366
    current best value =  692
    total weight =  373

    current total value =  336
    current best value =  692
    total weight =  292

    current total value =  393
    current best value =  692
    total weight =  312

    current total value =  336
    current best value =  692
    total weight =  292

    current total value =  418
    current best value =  692
    total weight =  392

    current total value =  347
    current best value =  692
    total weight =  331

    current total value =  276
    current best value =  692
    total weight =  285

    current total value =  194
    current best value =  692
    total weight =  185

    current total value =  239
    current best value =  692
    total weight =  264

    current total value =  336
    current best value =  692
    total weight =  321

    current total value =  418
    current best value =  692
    total weight =  421

    current total value =  412
    current best value =  692
    total weight =  413

    current total value =  441
    current best value =  692
    total weight =  504

    current total value =  530
    current best value =  692
    total weight =  535

    current total value =  433
    current best value =  692
    total weight =  478

    current total value =  388
    current best value =  692
    total weight =  399

    current total value =  359
    current best value =  692
    total weight =  308

    current total value =  365
    current best value =  692
    total weight =  316

    current total value =  363
    current best value =  692
    total weight =  288

    current total value =  408
    current best value =  692
    total weight =  367

    current total value =  468
    current best value =  692
    total weight =  439

    current total value =  525
    current best value =  692
    total weight =  459

    current total value =  500
    current best value =  692
    total weight =  416

    current total value =  410
    current best value =  692
    total weight =  361

    current total value =  500
    current best value =  692
    total weight =  416

    current total value =  410
    current best value =  692
    total weight =  361

    current total value =  440
    current best value =  692
    total weight =  442

    current total value =  531
    current best value =  692
    total weight =  526

    current total value =  482
    current best value =  692
    total weight =  494

    current total value =  579
    current best value =  692
    total weight =  551

    current total value =  573
    current best value =  692
    total weight =  543

    current total value =  602
    current best value =  692
    total weight =  634

    current total value =  604
    current best value =  692
    total weight =  662

    current total value =  547
    current best value =  692
    total weight =  642

    current total value =  612
    current best value =  692
    total weight =  702

    current total value =  590
    current best value =  692
    total weight =  683

    current total value =  499
    current best value =  692
    total weight =  599

    current total value =  521
    current best value =  692
    total weight =  618

    current total value =  456
    current best value =  692
    total weight =  558

    current total value =  547
    current best value =  692
    total weight =  642

    current total value =  607
    current best value =  692
    total weight =  651

    current total value =  632
    current best value =  692
    total weight =  694

    current total value =  602
    current best value =  692
    total weight =  613

    current total value =  505
    current best value =  692
    total weight =  556

    current total value =  524
    current best value =  692
    total weight =  583

    current total value =  502
    current best value =  692
    total weight =  564

    current total value =  573
    current best value =  692
    total weight =  625

    current total value =  491
    current best value =  692
    total weight =  525

    current total value =  489
    current best value =  692
    total weight =  497

    current total value =  398
    current best value =  692
    total weight =  413

    current total value =  309
    current best value =  692
    total weight =  382

    current total value =  284
    current best value =  692
    total weight =  339

    current total value =  375
    current best value =  692
    total weight =  423

    current total value =  405
    current best value =  692
    total weight =  504

    current total value =  454
    current best value =  692
    total weight =  536

    current total value =  519
    current best value =  692
    total weight =  596

    current total value =  590
    current best value =  692
    total weight =  642

    current total value =  561
    current best value =  692
    total weight =  551

    current total value =  490
    current best value =  692
    total weight =  505

    current total value =  430
    current best value =  692
    total weight =  496

    current total value =  436
    current best value =  692
    total weight =  504

    current total value =  526
    current best value =  692
    total weight =  559

    current total value =  623
    current best value =  692
    total weight =  616

    current total value =  532
    current best value =  692
    total weight =  532

    current total value =  472
    current best value =  692
    total weight =  460

    current total value =  497
    current best value =  692
    total weight =  503

    current total value =  452
    current best value =  692
    total weight =  424

    current total value =  541
    current best value =  692
    total weight =  455

    current total value =  476
    current best value =  692
    total weight =  395

    current total value =  558
    current best value =  692
    total weight =  495

    current total value =  468
    current best value =  692
    total weight =  440

    current total value =  443
    current best value =  692
    total weight =  397

    current total value =  394
    current best value =  692
    total weight =  365

    current total value =  459
    current best value =  692
    total weight =  425

    current total value =  550
    current best value =  692
    total weight =  509

    current total value =  461
    current best value =  692
    total weight =  478

    current total value =  490
    current best value =  692
    total weight =  569

    current total value =  561
    current best value =  692
    total weight =  615

    current total value =  464
    current best value =  692
    total weight =  558

    current total value =  458
    current best value =  692
    total weight =  550

    current total value =  376
    current best value =  692
    total weight =  450

    current total value =  436
    current best value =  692
    total weight =  459

    current total value =  365
    current best value =  692
    total weight =  398

    current total value =  335
    current best value =  692
    total weight =  317

    current total value =  365
    current best value =  692
    total weight =  398

    current total value =  462
    current best value =  692
    total weight =  455

    current total value =  522
    current best value =  692
    total weight =  527

    current total value =  528
    current best value =  692
    total weight =  535

    current total value =  468
    current best value =  692
    total weight =  526

    current total value =  525
    current best value =  692
    total weight =  546

    current total value =  614
    current best value =  692
    total weight =  577

    current total value =  616
    current best value =  692
    total weight =  605

    current total value =  687
    current best value =  692
    total weight =  666

    current total value =  616
    current best value =  692
    total weight =  620

    current total value =  638
    current best value =  692
    total weight =  639

    current total value =  663
    current best value =  692
    total weight =  682

    current total value =  723
    current best value =  692
    total weight =  691

    current total value =  794
    current best value =  692
    total weight =  737

    current total value =  737
    current best value =  692
    total weight =  717

    current total value =  666
    current best value =  692
    total weight =  656

    current total value =  595
    current best value =  692
    total weight =  610

    current total value =  677
    current best value =  692
    total weight =  710

    current total value =  586
    current best value =  692
    total weight =  626

    current total value =  526
    current best value =  692
    total weight =  617

    current total value =  586
    current best value =  692
    total weight =  626

    current total value =  526
    current best value =  692
    total weight =  617

    current total value =  501
    current best value =  692
    total weight =  574

    current total value =  419
    current best value =  692
    total weight =  474

    current total value =  464
    current best value =  692
    total weight =  553

    current total value =  524
    current best value =  692
    total weight =  562

    current total value =  549
    current best value =  692
    total weight =  605

    current total value =  484
    current best value =  692
    total weight =  545

    current total value =  465
    current best value =  692
    total weight =  518

    current total value =  436
    current best value =  692
    total weight =  427

    current total value =  465
    current best value =  692
    total weight =  518

    current total value =  435
    current best value =  692
    total weight =  437

    current total value =  375
    current best value =  692
    total weight =  365

    current total value =  373
    current best value =  692
    total weight =  337

    current total value =  284
    current best value =  692
    total weight =  306

    current total value =  366
    current best value =  692
    total weight =  406

    current total value =  321
    current best value =  692
    total weight =  327

    current total value =  411
    current best value =  692
    total weight =  382

    current total value =  482
    current best value =  692
    total weight =  428

    current total value =  542
    current best value =  692
    total weight =  500

    current total value =  587
    current best value =  692
    total weight =  579

    current total value =  516
    current best value =  692
    total weight =  533

    current total value =  573
    current best value =  692
    total weight =  553

    current total value =  513
    current best value =  692
    total weight =  544

    current total value =  604
    current best value =  692
    total weight =  628

    current total value =  507
    current best value =  692
    total weight =  571

    current total value =  447
    current best value =  692
    total weight =  499

    current total value =  402
    current best value =  692
    total weight =  420

    current total value =  473
    current best value =  692
    total weight =  481

    current total value =  522
    current best value =  692
    total weight =  513

    current total value =  582
    current best value =  692
    total weight =  585

    current total value =  522
    current best value =  692
    total weight =  513

    current total value =  524
    current best value =  692
    total weight =  541

    current total value =  613
    current best value =  692
    total weight =  572

    current total value =  710
    current best value =  692
    total weight =  629

    current total value =  621
    current best value =  692
    total weight =  598

    current total value =  615
    current best value =  692
    total weight =  590

    current total value =  645
    current best value =  692
    total weight =  671

    current total value =  690
    current best value =  692
    total weight =  750

    current total value =  688
    current best value =  692
    total weight =  722

    current total value =  759
    current best value =  692
    total weight =  768

    current total value =  729
    current best value =  692
    total weight =  687

    current total value =  818
    current best value =  692
    total weight =  718

    current total value =  837
    current best value =  692
    total weight =  745

    current total value =  902
    current best value =  692
    total weight =  805

    current total value =  853
    current best value =  692
    total weight =  773

    current total value =  834
    current best value =  692
    total weight =  746

    current total value =  894
    current best value =  692
    total weight =  818

    current total value =  805
    current best value =  692
    total weight =  787

    current total value =  734
    current best value =  692
    total weight =  741

    current total value =  705
    current best value =  692
    total weight =  650

    current total value =  794
    current best value =  692
    total weight =  681

    current total value =  769
    current best value =  692
    total weight =  638

    current total value =  698
    current best value =  692
    total weight =  577

    current total value =  704
    current best value =  692
    total weight =  585

    current total value =  644
    current best value =  692
    total weight =  513

    current total value =  547
    current best value =  692
    total weight =  456

    current total value =  525
    current best value =  692
    total weight =  437

    current total value =  460
    current best value =  692
    total weight =  377

    current total value =  485
    current best value =  692
    total weight =  420

    current total value =  479
    current best value =  692
    total weight =  412

    current total value =  539
    current best value =  692
    total weight =  484

    current total value =  568
    current best value =  692
    total weight =  575

    current total value =  574
    current best value =  692
    total weight =  583

    current total value =  517
    current best value =  692
    total weight =  563

    current total value =  511
    current best value =  692
    total weight =  555

    current total value =  576
    current best value =  692
    total weight =  615

    current total value =  598
    current best value =  692
    total weight =  634

    current total value =  655
    current best value =  692
    total weight =  654

    current total value =  752
    current best value =  692
    total weight =  711

    current total value =  782
    current best value =  692
    total weight =  792

    current total value =  725
    current best value =  692
    total weight =  772

    current total value =  628
    current best value =  692
    total weight =  715

    current total value =  688
    current best value =  692
    total weight =  724

    current total value =  643
    current best value =  692
    total weight =  645

    current total value =  614
    current best value =  692
    total weight =  554

    current total value =  554
    current best value =  692
    total weight =  545

    current total value =  573
    current best value =  692
    total weight =  572

    current total value =  618
    current best value =  692
    total weight =  651

    current total value =  667
    current best value =  692
    total weight =  683

    current total value =  602
    current best value =  692
    total weight =  623

    current total value =  553
    current best value =  692
    total weight =  591

    current total value =  531
    current best value =  692
    total weight =  572

    current total value =  628
    current best value =  692
    total weight =  629

    current total value =  603
    current best value =  692
    total weight =  586

    current total value =  652
    current best value =  692
    total weight =  618

    current total value =  717
    current best value =  692
    total weight =  678

    current total value =  742
    current best value =  692
    total weight =  721

    current total value =  697
    current best value =  692
    total weight =  642

    current total value =  672
    current best value =  692
    total weight =  599

    current total value =  717
    current best value =  692
    total weight =  678

    current total value =  687
    current best value =  692
    total weight =  597

    current total value =  758
    current best value =  692
    total weight =  643

    current total value =  661
    current best value =  692
    total weight =  586

    current total value =  732
    current best value =  692
    total weight =  647

    current total value =  713
    current best value =  692
    total weight =  620

    current total value =  653
    current best value =  692
    total weight =  548

    current total value =  713
    current best value =  692
    total weight =  620

    current total value =  738
    current best value =  692
    total weight =  663

    current total value =  649
    current best value =  692
    total weight =  632

    current total value =  678
    current best value =  692
    total weight =  723

    current total value =  767
    current best value =  692
    total weight =  754

    current total value =  722
    current best value =  692
    total weight =  675

    current total value =  697
    current best value =  692
    total weight =  632

    current total value =  648
    current best value =  692
    total weight =  600

    current total value =  705
    current best value =  692
    total weight =  620

    current total value =  614
    current best value =  692
    total weight =  536

    current total value =  620
    current best value =  692
    total weight =  544

    current total value =  650
    current best value =  692
    total weight =  625

    current total value =  699
    current best value =  692
    total weight =  657

    current total value =  628
    current best value =  692
    total weight =  596

    current total value =  557
    current best value =  692
    total weight =  550

    current total value =  579
    current best value =  692
    total weight =  569

    current total value =  530
    current best value =  692
    total weight =  537

    current total value =  524
    current best value =  692
    total weight =  529

    current total value =  584
    current best value =  692
    total weight =  538

    current total value =  524
    current best value =  692
    total weight =  529

    current total value =  569
    current best value =  692
    total weight =  608

    current total value =  504
    current best value =  692
    total weight =  548

    current total value =  482
    current best value =  692
    total weight =  529

    current total value =  392
    current best value =  692
    total weight =  474



*last editted: 01/06/19*

<a href="#top">Go to top</a>
