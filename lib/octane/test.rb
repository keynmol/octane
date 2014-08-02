module Octane
module Test
	def test_regression(dataset)
			dataset=dataset.data if dataset.is_a?(Dataset)
			test_error=0.0
			dataset.each{|example,output|
				result=forward_pass(example)
				test_error+=result.zip(output).map{|r, o| (r-o)**2}.reduce(:+)
			}
			test_error/(2*dataset.length)
		end

		def test_classification(dataset)
			correctly_classified=0.0
			dataset.each{|example, output|
				result=forward_pass(example)
				computed=result.index(result.max)
				expected=output.index(output.max)
				correctly_classified+=1 if computed==expected
			}

			{correct: correctly_classified/dataset.size}
		end

		def test(dataset)
			if dataset.is_a?(ClassificationDataset)
				test_classification(dataset)
			else
				test_regression dataset
			end
		end
end
end