module Octane
module Training
def train_defaults(options)
			options[:epochs]=options[:epochs] || 10

			options
		end

		def train(dataset, options={})
			options=train_defaults(options)
			dataset=dataset.data if dataset.is_a?(Dataset)

			options[:test_set]=options[:test_set].data if options[:test_set] && options[:test_set].is_a?(Dataset)
			batch_learning=!!options[:batch]
			
			options[:iterations]=dataset.size/options[:batch] if batch_learning && options[:iterations].nil?
			options[:iterations]||=dataset.size

			

			options[:iterations].times do |timestep|
				
				example,expected_output=dataset.sample

				options[:before_training][self, timestep, example, expected_output] if options[:before_training]

				train_one([example,expected_output], options)
			end
		end

		def train_one(sample, options={})
			example, expected_output=sample
			options[:epochs]||=1
			example_error=0
			options[:epochs].times do |epoch|
				network_output=forward_pass(example)
				example_error=calculate_errors(network_output, expected_output)
				calculate_weight_changes
				apply_weight_changes 1
			end
			example_error
		end

		def train_batch(batch, options={})
			batch.each do |example, expected_output|
				network_output=forward_pass(example)
				example_error=calculate_errors(network_output, expected_output)
				calculate_weight_changes
			end

			apply_weight_changes batch.size
		end
end
end