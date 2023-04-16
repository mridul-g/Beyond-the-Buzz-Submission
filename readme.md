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
                        <li>Run the code.</li>
                    </ol>
                    <h2>Working</h2>
                        <ol>
                            <li>The code loads the data and prints the numerical variables in the data and a histogram for each of them.</li>
                            <li>It then splits the data into train and test sets and trains different models, such as Random Forest, Gradient Boosting, SVM, and Logistic Regression, and calculates their accuracy scores.</li>
                            <li>Finally, it selects the best performing model (Random Forest) and predicts the verdict for the test data and saves it to a file named predictions.csv.</li>
                        </ol>                    
                    <h2>Dataset</h2>
                    <p>The dataset used in this code contains various features related to a legal case, such as the location of the incident, the type of crime, and the age and gender of the suspect. The dataset also includes a binary target variable "VERDICT" indicating whether the suspect was found guilty or not guilty.</p>
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