---
permalink: /Text-Classification/
header:
  image: "/images/digital-transition2.jpg"
---


### Multinomial Logit Regression for Twitter Text Classification

In this notebook, we will apply logistic regression to classify twitter text into three tones which are "neutral", "offensive language" and "hate speech".

First of all, let's import library that we will use in this note book.

### import library


```R
options(warn=-1) #turn off warning
options(jupyter.plot_mimetypes = 'image/png')
library('glmnet')
library('RTextTools')
```

    Loading required package: Matrix
    Loading required package: foreach
    Loaded glmnet 2.0-5

    Loading required package: SparseM

    Attaching package: 'SparseM'

    The following object is masked from 'package:base':

        backsolve



### read data in

data that will be used in this notebook are comprised of training data and test data.

Now, let's read data in from CSV file.


```R
df <- read.csv("Twitter-hate_speech-labeled_data - Sheet1.csv")
test <- read.csv("Twitter-hate_speech-test_unlabeled - Sheet1.csv")
head(df)
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>tweet_text</th><th scope=col>Label</th></tr></thead>
<tbody>
	<tr><td> 2                                                                                                                                                   </td><td>@NFLfantasy @Akbar_Gbaja Damn right, Akbar.                                                                                                          </td><td>neutral                                                                                                                                              </td></tr>
	<tr><td> 3                                                                                                                                                   </td><td>@NFLfantasy @Akbar_Gbaja I've got backup _â€°Ð³Ñžâ€°Ð«__                                                                                             </td><td>neutral                                                                                                                                              </td></tr>
	<tr><td> 5                                                                                                                                                   </td><td>"No one is going to replace the production of Odell Beckham Jr. He's let a lot of fantasy owners down."-@Akbar_Gbaja https://t.co/8Nd1Knzunq         </td><td>neutral                                                                                                                                              </td></tr>
	<tr><td> 7                                                                                                                                                   </td><td>@Sheikh__Akbar It's actually PERFECT for you man! Always on the phone_â€°Ð³Ñžâ€°Ð«_Ðœ__â€°Ð³Ñžâ€°Ð«_Ðœ__â€°Ð³Ñžâ€°Ð«_Ðœ_                             </td><td>neutral                                                                                                                                              </td></tr>
	<tr><td>10                                                                                                                                                   </td><td>Was Trump right? Officers say pockets of Muslims celebrated 9/11 https://t.co/NSVY7rDwcu via @MailOnline                                             </td><td>neutral                                                                                                                                              </td></tr>
	<tr><td>12                                                                                                                                                   </td><td>Female Killer In Las Vegas Shouted ÐƒÐšâ€°Ð«Ð£â€°Ð«_Allahu AkbarÐƒÐšâ€°Ð«Ð£ÐœÑ† As She Ran Over 40 Innocent People Last Night https://t.co/wjYGYzRCmD</td><td>neutral                                                                                                                                              </td></tr>
</tbody>
</table>



Now, let's inspect classes that text is classified into by using unique command in R which will tell us how many and what classes there are in the training set.


```R
unique(df$Label) #this will print out 3 classes that we will have to classify later.
```


<ol class=list-inline>
	<li>neutral</li>
	<li>offensive language</li>
	<li>hate speech</li>
</ol>



Let's print out the number of training dataset and test dataset.


```R
print(nrow(df))
print(nrow(test))
```

    [1] 11418
    [1] 3024


### merge df and test tweet text and create corpus

Now, we will bind train data and test data in order to apply corpus transformation latter on.


```R
data <- rbind(df, test)
print(nrow(data))
```

    [1] 14442


### use 'RTextTools' for transformation to corpus and remove stop words

Now we will make the word corpus and also remove stopwords as well as number from our corpus.


```R
# prepare data
corpus <- create_matrix(data$tweet_text, language = "english", removeStopwords = TRUE,
    removeNumbers = TRUE, stemWords = FALSE, tm::weightTfIdf)
corpusmatrix <- as.matrix(corpus)
```

Now, we will randomly split training data into training set and validation set to use to validate our model.


```R
train_ind <- sample(seq_len(nrow(df)), size = floor(0.75 * nrow(df)))
dfmatrix <- corpusmatrix[1:nrow(df),]
trainmatrix <- dfmatrix[train_ind, ]
trainlabel <- as.factor(df$Label[train_ind])
validmatrix <- dfmatrix[-train_ind, ]
validlabel <- as.factor(df$Label[-train_ind])
```

### train 'glmnet' multinomial

Now, we train logit regression with losso regularisation. In the meanwhile, this will also perform variable elimination from the model as well.


```R
multifit2 <- cv.glmnet(trainmatrix, trainlabel, family="multinomial", type.multinomial = "grouped", parallel = TRUE)
```

### plot fit

The following plot visualise the value of log lambda that gives the least error.


```R
plot(multifit2)
```


![png](output_24_0.png)


### validate model

We will now validate the performance of our model by using validation set that we put aside earlier.


```R
pvalid2 <- predict(multifit2, validmatrix , s="lambda.min", type="class")
```

### confusion metrix

After we apply our model to the validation set, now let's see how well it can classify our validation set. This can be done using command table which will create "confusion metrix" for us automatically.


```R
predicted_df <- as.data.frame(cbind(as.character(pvalid2), as.character(validlabel)))
colnames(predicted_df) <- c("predicted","actual")
table(predicted_df)
```


                        actual
    predicted            hate speech neutral offensive language
      hate speech                169       0                112
      neutral                     63    1357                151
      offensive language         278      40                685


### predict classes

After all the hustle, let's apply the model that we build to our test set. Then, let's inspect first 10 elements in our test set.


```R
pfit2 <- predict(multifit2, corpusmatrix[(nrow(df)+1):nrow(corpusmatrix),] , s="lambda.min", type="class")
test$Label <- pfit2
head(test)
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>tweet_text</th><th scope=col>Label</th></tr></thead>
<tbody>
	<tr><td> 1                                                                                                                       </td><td>@NFLfantasy @Akbar_Gbaja the last thing he cares about is fantasy owners                                                 </td><td>neutral                                                                                                                  </td></tr>
	<tr><td> 4                                                                                                                       </td><td>@NFLfantasy @Akbar_Gbaja He also let his team down. It's a must win game.                                                </td><td>neutral                                                                                                                  </td></tr>
	<tr><td> 6                                                                                                                       </td><td>Man Shouting 'Allahu Akbar' Drives Into Crowd https://t.co/pznvkur9cT                                                    </td><td>neutral                                                                                                                  </td></tr>
	<tr><td> 8                                                                                                                       </td><td>Nan _â€°Ð³Ñžâ€°Ð«Ñžâ€°Ð«Ñž (@ Akbar's in Leeds, West Yorkshire) https://t.co/9HbyQorVmO https://t.co/Gv6AXUSoiA          </td><td>neutral                                                                                                                  </td></tr>
	<tr><td> 9                                                                                                                       </td><td>VIDEOS: Immigrant Somali Muslim Woman Yelling Allahu Akbar Intentionally Ran Over 40 People On... https://t.co/DYlGZKOFjw</td><td>neutral                                                                                                                  </td></tr>
	<tr><td>11                                                                                                                       </td><td>Was Trump right? Officers say pockets of Muslims celebrated 9/11 https://t.co/ZQLM9XYtKN via @MailOnline                 </td><td>neutral                                                                                                                  </td></tr>
</tbody>
</table>




```R
#get some samples of "hate speech"
head(subset(test, Label=="hate speech",))
```


<table>
<thead><tr><th></th><th scope=col>id</th><th scope=col>tweet_text</th><th scope=col>Label</th></tr></thead>
<tbody>
	<tr><th scope=row>162</th><td>842                                                                                      </td><td>@rwnc70 @JebBush LOL suck my motherfucking dick, faggot                                  </td><td>hate speech                                                                              </td></tr>
	<tr><th scope=row>165</th><td>872                                                                                      </td><td>@Alexcisneros69 Shut up you fucking faggot.                                              </td><td>hate speech                                                                              </td></tr>
	<tr><th scope=row>169</th><td>886                                                                                                                                                          </td><td><span style=white-space:pre-wrap>running with lfg faggots who think they know everything &amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;     </span></td><td>hate speech                                                                                                                                                  </td></tr>
	<tr><th scope=row>170</th><td>902                                                                                      </td><td>Bunch of slack jawed faggots around here! https://t.co/wmqqqFCYL9 https://t.co/t0OdmYF9gH</td><td>hate speech                                                                              </td></tr>
	<tr><th scope=row>171</th><td>903                                                                                      </td><td>@kalumevs @CallumHarries I've seen you two bench so don't pipe up #Faggots               </td><td>hate speech                                                                              </td></tr>
	<tr><th scope=row>172</th><td>906                                                                                      </td><td>Faggots? https://t.co/ldAAHEC8gO                                                         </td><td>hate speech                                                                              </td></tr>
</tbody>
</table>




```R
#get some samples of "offensive language"
head(subset(test, Label=="offensive language",))
```


<table>
<thead><tr><th></th><th scope=col>id</th><th scope=col>tweet_text</th><th scope=col>Label</th></tr></thead>
<tbody>
	<tr><th scope=row>11</th><td> 42                                                                                                                                                                                                                        </td><td>Dem philly niggas b so great at that dressing them some fly niggas they make everything they put on look great _â€°Ð³Ñžâ€°Ð«_ÐµÐŒ_â€°Ð³Ñžâ€°Ð«_ÐµÐŒ_â€°Ð³Ñžâ€°Ð«_ÐµÐŒ_â€°Ð³Ñžâ€°Ð«_ÐµÐŒ_â€°Ð³Ñžâ€°Ð«_ÐµÐŒ_â€°Ð³Ñžâ€°Ð«_ÐµÐ</td><td>offensive language                                                                                                                                                                                                         </td></tr>
	<tr><th scope=row>40</th><td>182                                                                                                                                                                                                                        </td><td>The same thing applies to these niggas too. How you gone get mad a bitch posted a picture of yo tiny ashy ass dick?                                                                                                        </td><td>offensive language                                                                                                                                                                                                         </td></tr>
	<tr><th scope=row>42</th><td>188                                                                                                                                                                                                                        </td><td>When you fuck darkskin girls in the winter time they ass cheeks Be ashy as hell _â€°Ð³Ñžâ€°Ð«_Ðœ__â€°Ð³Ñžâ€°Ð«_Ðœ__â€°Ð³Ñžâ€°Ð«_Ðœ_                                                                                        </td><td>offensive language                                                                                                                                                                                                         </td></tr>
	<tr><th scope=row>45</th><td>208                                                                                                                                                                                                                        </td><td>@EgyptTaughtMe @srslyab that's not even your skin tone and that's not even your body you ashy bitch https://t.co/TwBW9O9wvx                                                                                                </td><td>offensive language                                                                                                                                                                                                         </td></tr>
	<tr><th scope=row>47</th><td>226                                                                                                                                                                                                                        </td><td>Nothing says "IÐƒÐšâ€°Ð«Ð£ÐœÑ†m a fat bastard" like wearing a T-shirt in a swimming pool.                                                                                                                                  </td><td>offensive language                                                                                                                                                                                                         </td></tr>
	<tr><th scope=row>50</th><td>246                                                                                                                                                                                                                        </td><td>That monochrome camera is brand new and costs about six grand. Bastard.                                                                                                                                                    </td><td>offensive language                                                                                                                                                                                                         </td></tr>
</tbody>
</table>



*last edit 25/10/2016*
