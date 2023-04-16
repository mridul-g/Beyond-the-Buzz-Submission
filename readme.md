<div class="relative flex w-[calc(100%-50px)] flex-col gap-1 md:gap-3 lg:w-[calc(100%-115px)]">
    <div class="flex flex-grow flex-col gap-3">
        <div class="min-h-[20px] flex flex-col items-start gap-4 whitespace-pre-wrap">
            <div class="markdown prose w-full break-words dark:prose-invert light">
                <h1>Readme for Beyond the Buzz recruitment task Submission</h1>
                    <p>This code is a machine learning model that uses various classifiers to predict the verdict of a legal case based on input features.</p>
                    <h2>Installation</h2>
                    <p>To run this code, you need to install Python 3.5 or above and the following packages:</p>
                    <ul>
                        <li>
                            <code>numpy</code>: A Python library for numerical computing</li>
                        <li><code>pandas</code>: A Python library for data manipulation</li>
                        <li><code>matplotlib</code>: A Python library for data visualization</li>
                        <li><code>seaborn</code>: A Python library for statistical data visualization</li>
                        <li><code>scikit-learn</code>: A machine learning library for Python</li>
                    </ul>
                    <p>You can install the required packages using <code>pip</code> by running the following command:</p>
                    <div class="codeHeader" id="code-try-0" data-bi-name="code-header">
                		<button type="button" class="action position-relative display-none-print" data-bi-name="copy">
                			<div class="successful-copy-alert position-absolute right-0 top-0 left-0 bottom-0 display-flex align-items-center justify-content-center has-text-success-invert has-background-success is-transparent" aria-hidden="true">
                				<span class="icon font-size-lg">
                					<span class="docon docon-check-mark"></span>
                				</span>
                			</div>
                		</button>
                	</div>
                    <pre tabindex="0" class="has-inner-focus ml-auto">
                        <code class="lang-console" data-author-content="pip install numpy pandas matplotlib seaborn scikit-learn">
                            <span>pip install numpy pandas matplotlib seaborn scikit-learn</span>
                        </code>
                    </pre>
                    <h2>Usage</h2>
                    <p>The code reads in training and test datasets and applies various classifiers (Random Forest, Gradient Boosting, SVM, and Logistic Regression) to predict the verdict of a legal case based on input features. The predictions are then saved to a CSV file for submission.</p>
                    <p>To use this code, you need to follow the steps below:</p>
                    <ol>
                        <li>Ensure that the required packages are installed (see Installation section).</li>
                        <li>Download the training and test datasets (in CSV format) and save them in the "data" directory with the names "train.csv" and "test.csv", respectively.</li>
                        <li>Run the code. </li>
                    </ol>
                    <h2>Working</h2>
                        <ol>
                            <li>The code loads the data and prints the numerical variables in the data and a histogram for each of them.</li>
                            <li>It then splits the data into train and test sets and trains different models, such as Random Forest, Gradient Boosting, SVM, and Logistic Regression, and calculates their accuracy scores.</li>
                            <li>Finally, it selects the best performing model (Random Forest) and predicts the verdict for the test data and saves it to a file named predictions.csv.</li>
                        </ol>
                    <h2>Random Forest Classifier</h2>
                    <p>Random Forest is a popular ensemble learning method for classification, regression, and other tasks. It is an extension of decision trees, where multiple decision trees are trained on randomly selected subsets of the training data and features.</p>
                    <p>Here's how Random Forest Classifier works: </p>
                        <ol>                  
                            <li>Selecting random samples from a given dataset.</li>
                            <li>Building a decision tree for each sample and getting a prediction result from each decision tree.</li>
                            <li>Performing a vote for each predicted result.</li>
                            <li>Selecting the prediction result with the most votes as the final prediction.</li>
                        </ol>
                    <p>The key concept behind Random Forest is to combine the outputs of multiple decision trees to create a more accurate and stable prediction. This helps to reduce overfitting and improve generalization performance.</p>
                    <p>During training, the Random Forest algorithm creates multiple decision trees by randomly selecting subsets of data and features. The number of trees and the size of the subsets are hyperparameters that can be tuned for optimal performance.</p>
                    <p>During prediction, each decision tree in the Random Forest makes a prediction, and the class with the most votes across all trees is selected as the final prediction.</p>
                    <p>The Random Forest Classifier is based on two key concepts: bagging and random feature selection.</p>
                    <h3>Bagging (Bootstrap Aggregating):</h3>
                    <p>Bagging is a technique that involves sampling the training data with replacement to create multiple subsets of the data. The decision tree is then trained on each of these subsets, and the final prediction is obtained by combining the predictions of all decision trees. Bagging helps to reduce overfitting and improve the stability of the model.</p>
                    <h3>Random Feature Selection:</h3>
                    <p>Random feature selection is a technique that involves randomly selecting a subset of features for each decision tree. This helps to reduce the correlation between decision trees and ensures that each decision tree makes a different set of decisions.</p>
                    <p>The math behind the Random Forest Classifier algorithm involves constructing decision trees and aggregating their predictions using majority voting. Each decision tree is constructed recursively by selecting the best feature to split the data at each node based on a metric such as information gain or Gini impurity. The process continues until the data is fully partitioned, or a stopping criterion is met. The final prediction is obtained by aggregating the predictions of all decision trees using majority voting.</p>
                    <h2>Dataset</h2>
                    <p>The dataset used in this code contains various features related to a legal case, such as the location of the incident, the type of crime, and the age and gender of the suspect. The dataset also includes a binary target variable "VERDICT" indicating whether the suspect was found guilty or not guilty.</p>
                    <p>Finally the predictions were made using Random Forest Classifier with an accuracy of 94.8734 %</p>
                </div>
            </div>
        </div>
        <div class="flex justify-between lg:block">
            <div class="text-gray-400 flex self-end lg:self-center justify-center mt-2 gap-2 md:gap-3 lg:gap-1 lg:absolute lg:top-0 lg:translate-x-full lg:right-0 lg:mt-0 lg:pl-2 visible">
                <button class="p-1 rounded-md hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-gray-200 disabled:dark:hover:text-gray-400">
                    <svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>
                </button>
                <button class="p-1 rounded-md hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-gray-200 disabled:dark:hover:text-gray-400"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path></svg></button>
            </div>
        </div>
    </div>
</div>