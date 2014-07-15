module Octane
	class Dataset
		def initialize
			@dataset=[]
		end

		def add_sample(sample,expected)
			@dataset<<[sample,expected.is_a?(Array) ? expected : [expected]]
		end

		def data
			@dataset
		end

		def batch(size)
			@dataset.sample(size)
		end

		def data=(data)
			@dataset=data
		end

		def clear
			@dataset=[]
		end

		def clone
			new_dataset=Dataset.new
			new_dataset.data=Marshal.load(Marshal.dump(@dataset.clone)) # TODO: god that's ugly.
			new_dataset
		end
	end
end