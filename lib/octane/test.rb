module Octane
module Test
	def test_regression(dataset, options={})
			dataset=dataset.data if dataset.is_a?(Dataset)
			test_error=0.0
			dataset.each{|example,output|
				example=@input_transformation[example] if @input_transformation
				result=forward_pass(example)
				result=options[:output_transformation].call(result) if options[:output_transformation]
				
				raise "Expected output and network output have different dimensionalities!" if result.size!=output.size
				
				test_error+=result.zip(output).map{|r, o| (r-o)**2}.reduce(:+)/result.length
			}
			test_error/(2*dataset.length)
		end

		def test_classification(dataset, options={})
			correctly_classified=0.0
			dataset.each{|example, output|
				example=@input_transformation[example] if @input_transformation
				result=forward_pass(example)
				computed=result.index(result.max)
				expected=output.index(output.max)
				correctly_classified+=1 if computed==expected
			}

			{correct: correctly_classified/dataset.size}
		end

		def test(dataset, options={})
			if dataset.is_a?(ClassificationDataset)
				test_classification(dataset, options)
			else
				test_regression dataset, options
			end
		end
end
end